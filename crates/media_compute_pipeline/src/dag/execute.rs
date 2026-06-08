//! # Execution — parallel DAG scheduling with topological level barriers.
//!
//! For each topological depth level, all independent nodes execute
//! concurrently via [`rayon::scope`].  The caller supplies an
//! [`EffectProvider`](crate::EffectProvider) that resolves effect kinds
//! to actual computation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use media_compute::EffectDesc;
use media_core::error::{MediaError, MediaResult};
use media_core::Frame;

use super::{ComputeDag, DagResult, DagStats, NodeStat};
use crate::pool::FramePool;

// ---------------------------------------------------------------------------
// apply_dag (allocating)
// ---------------------------------------------------------------------------

/// Execute a DAG without a frame pool (allocates fresh frames for every node).
pub(crate) fn apply_dag(
    dag: &ComputeDag,
    input: &Frame,
    effects: &dyn crate::EffectProvider,
) -> MediaResult<DagResult> {
    let mut pool = FramePool::new();
    apply_dag_with_pool(dag, input, effects, &mut pool)
}

// ---------------------------------------------------------------------------
// apply_dag_with_pool (reuses allocations)
// ---------------------------------------------------------------------------

/// Execute a DAG using a frame pool for buffer reuse.
pub(crate) fn apply_dag_with_pool(
    dag: &ComputeDag,
    input: &Frame,
    effects: &dyn crate::EffectProvider,
    pool: &mut FramePool,
) -> MediaResult<DagResult> {
    let nodes = dag.nodes();
    let sorted = &dag.sorted;
    if sorted.is_empty() {
        return Ok(DagResult {
            outputs: HashMap::new(),
            stats: DagStats::empty(),
        });
    }

    let depth = &dag.depth;
    let max_depth = *depth.iter().max().unwrap_or(&0);

    // --- group indices by depth level -------------------------------------
    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for &idx in sorted {
        levels[depth[idx]].push(idx);
    }

    // --- buffer pool (thread-safe) ----------------------------------------
    let buffer: Arc<Mutex<HashMap<String, Frame>>> = Arc::new(Mutex::new(HashMap::new()));
    {
        let input_key = pool.input_key();
        buffer.lock().unwrap().insert(input_key, input.clone());
    }

    // --- per-node stats ---------------------------------------------------
    let stats: Arc<Mutex<Vec<NodeStat>>> = Arc::new(Mutex::new(Vec::with_capacity(nodes.len())));

    // --- error flag (set by any failing node) -----------------------------
    let failed: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let error_msg: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let start = Instant::now();

    // --- execute level by level -------------------------------------------
    for level_indices in &levels {
        if failed.load(Ordering::Acquire) {
            break;
        }

        rayon::scope(|scope| {
            for &idx in level_indices {
                if failed.load(Ordering::Acquire) {
                    break;
                }

                let node = &nodes[idx];
                let t0 = Instant::now();

                // Collect input frames by cloning from the buffer.
                let input_frames: Vec<Frame> = if node.depends_on.is_empty() {
                    let buf = buffer.lock().unwrap();
                    vec![buf.get(&pool.input_key()).unwrap().clone()]
                } else {
                    let buf = buffer.lock().unwrap();
                    node.depends_on
                        .iter()
                        .map(|dep| buf.get(dep).unwrap().clone())
                        .collect()
                };

                let buf_clone = Arc::clone(&buffer);
                let stats_clone = Arc::clone(&stats);
                let failed_clone = Arc::clone(&failed);
                let err_clone = Arc::clone(&error_msg);

                scope.spawn(move |_| {
                    if failed_clone.load(Ordering::Acquire) {
                        return;
                    }

                    let src = &input_frames[0];

                    let mut desc = EffectDesc::new(node.kind.clone());
                    for p in &node.params {
                        desc = desc.with(p.clone());
                    }

                    let result = effects.apply_effect(&node.kind, src, &desc);

                    match result {
                        Ok(frame) => {
                            let elapsed = t0.elapsed();
                            let mut buf = buf_clone.lock().unwrap();
                            buf.insert(node.name.clone(), frame);
                            stats_clone.lock().unwrap().push(NodeStat {
                                name: node.name.clone(),
                                kind: node.kind.clone(),
                                duration: elapsed,
                            });
                        }
                        Err(e) => {
                            failed_clone.store(true, Ordering::Release);
                            let mut em = err_clone.lock().unwrap();
                            *em = Some(format!(
                                "DAG node '{}' failed: {}",
                                node.name,
                                e.to_string()
                            ));
                        }
                    }
                });
            }
        });

        // Check for errors after this level completes.
        if let Some(msg) = error_msg.lock().unwrap().take() {
            return Err(MediaError::Other(msg));
        }
    }

    // --- Collect outputs --------------------------------------------------
    let final_buffer = buffer.lock().unwrap();
    let output_names = dag.output_names();
    let mut outputs = HashMap::with_capacity(output_names.len());
    for name in output_names {
        if let Some(frame) = final_buffer.get(name) {
            outputs.insert(name.to_string(), frame.clone());
        }
    }
    drop(final_buffer);

    let per_node = stats.lock().unwrap().clone();
    let total = start.elapsed();

    Ok(DagResult {
        outputs,
        stats: DagStats { per_node, total },
    })
}
