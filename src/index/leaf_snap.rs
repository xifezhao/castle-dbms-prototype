//! CASTLE Core: The Immutable Predictive Routing View (`LeafSnap`)
//!
//! Implements the read-only snapshot holding segmented Piecewise Linear Models (PLMs)
//! and a contiguous sorted base data array, 64-byte cache-line aligned for tagged-pointer
//! publication in `LeafCtrl`.

#![deny(missing_docs)]

use crate::index::plm::PiecewiseLinearModel;
use crate::{IndexKey, IndexValue};

/// Immutable snapshot holding the floating-point predictive models and sorted base run.
///
/// 64-byte alignment prevents false sharing and guarantees the lowest 6 bits of any
/// pointer are zero, enabling 2-bit tag packing (`TAG_BUF`, `TAG_REBUILD`) in `state_word`.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct LeafSnap<K: IndexKey, V: IndexValue> {
    /// Monotonic segment routing boundaries.
    boundaries: Box<[K]>,
    /// Piecewise Linear Models corresponding to each boundary segment.
    models: Box<[PiecewiseLinearModel]>,
    /// Contiguous sorted base array of committed key-value records.
    sorted_data: Box<[(K, V)]>,
}

impl<K: IndexKey, V: IndexValue> LeafSnap<K, V> {
    /// Constructs an empty valid `LeafSnap` for initial leaf bootstrap.
    pub fn empty() -> Self {
        Self {
            boundaries: vec![K::default()].into_boxed_slice(),
            models: vec![PiecewiseLinearModel::new(0.0, 0.0, 256)].into_boxed_slice(),
            sorted_data: Vec::new().into_boxed_slice(),
        }
    }

    /// Constructs a new immutable `LeafSnap` with empty sorted data.
    pub fn new(boundaries: Vec<K>, models: Vec<PiecewiseLinearModel>) -> Self {
        Self::new_with_data(boundaries, models, Vec::new())
    }

    /// Constructs a new immutable `LeafSnap` with an explicit sorted base array.
    pub fn new_with_data(
        boundaries: Vec<K>,
        models: Vec<PiecewiseLinearModel>,
        sorted_data: Vec<(K, V)>,
    ) -> Self {
        assert_eq!(
            boundaries.len(),
            models.len(),
            "Structural mismatch between routing boundaries and PLM segments."
        );
        Self {
            boundaries: boundaries.into_boxed_slice(),
            models: models.into_boxed_slice(),
            sorted_data: sorted_data.into_boxed_slice(),
        }
    }

    /// Returns a slice of the contiguous sorted base run.
    #[inline]
    pub fn sorted_data(&self) -> &[(K, V)] {
        &self.sorted_data
    }

    /// Returns the PLM models slice.
    #[inline]
    pub fn models(&self) -> &[PiecewiseLinearModel] {
        &self.models
    }

    /// Returns approximate heap memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.boundaries.len() * std::mem::size_of::<K>()
            + self.models.len() * std::mem::size_of::<PiecewiseLinearModel>()
            + self.sorted_data.len() * std::mem::size_of::<(K, V)>()
    }

    /// Evaluates the predictive model to estimate the offset of `key` in `sorted_data`.
    #[inline]
    pub fn predict(&self, key: &K) -> usize {
        if self.models.is_empty() {
            return 0;
        }
        let segment_idx = self.locate_segment(key);
        self.models[segment_idx].evaluate(key.to_f64())
    }

    /// Performs PLM-guided bounded search in `sorted_data`.
    #[inline]
    pub fn search(&self, key: &K, hard_max_error: usize) -> Option<V> {
        let len = self.sorted_data.len();
        if len == 0 {
            return None;
        }
        let pred = self.predict(key).min(len - 1);
        let lo = pred.saturating_sub(hard_max_error);
        let hi = (pred + hard_max_error + 1).min(len);

        // Fast path: binary search within [pred - delta_max, pred + delta_max]
        let slice = &self.sorted_data[lo..hi];
        if let Ok(idx) = slice.binary_search_by_key(key, |&(k, _)| k) {
            return Some(slice[idx].1.clone());
        }

        // Bounded fallback if key lies outside the predicted window
        if *key < slice[0].0 && lo > 0 {
            if let Ok(idx) = self.sorted_data[..lo].binary_search_by_key(key, |&(k, _)| k) {
                return Some(self.sorted_data[idx].1.clone());
            }
        } else if *key > slice[slice.len() - 1].0 && hi < len {
            if let Ok(idx) = self.sorted_data[hi..].binary_search_by_key(key, |&(k, _)| k) {
                return Some(self.sorted_data[hi + idx].1.clone());
            }
        }
        None
    }

    /// Performs PLM-guided lower-bound search and sequential scan into `out`.
    #[inline]
    pub fn range_scan(&self, low: &K, limit: usize, hard_max_error: usize, out: &mut Vec<(K, V)>) {
        let len = self.sorted_data.len();
        if len == 0 || limit == 0 {
            return;
        }
        let pred = self.predict(low).min(len - 1);
        let lo = pred.saturating_sub(hard_max_error);
        let hi = (pred + hard_max_error + 1).min(len);

        let start_idx = if *low <= self.sorted_data[lo].0 {
            self.sorted_data[..lo].partition_point(|&(k, _)| k < *low)
        } else if *low > self.sorted_data[hi - 1].0 {
            hi + self.sorted_data[hi..].partition_point(|&(k, _)| k < *low)
        } else {
            lo + self.sorted_data[lo..hi].partition_point(|&(k, _)| k < *low)
        };

        let end_idx = (start_idx + limit).min(len);
        out.extend_from_slice(&self.sorted_data[start_idx..end_idx]);
    }

    #[inline]
    fn locate_segment(&self, key: &K) -> usize {
        if self.boundaries.len() <= 1 {
            return 0;
        }
        match self.boundaries.binary_search(key) {
            Ok(exact_idx) => exact_idx,
            Err(insert_idx) => insert_idx.saturating_sub(1),
        }
    }
}

const _: () = assert!(
    std::mem::align_of::<LeafSnap<u64, u64>>() == 64,
    "LeafSnap MUST be cache-line aligned (64 bytes) for pointer tagging and false-sharing avoidance."
);
