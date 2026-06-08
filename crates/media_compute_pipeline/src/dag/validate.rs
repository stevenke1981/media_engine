//! # Validation — topological sort, cycle detection, and error checking.

use std::collections::{HashMap, VecDeque};

use media_core::error::{MediaError, MediaResult};

use super::DagNode;

/// Validated state produced by [`validate`].
#[derive(Debug, Clone)]
pub(crate) struct ValidateState {
    /// Indices sorted topologically.
    pub sorted: Vec<usize>,
    /// name → node index.
    pub name_map: HashMap<String, usize>,
    /// Topological depth for each node.
    pub depth: Vec<usize>,
    /// Reverse adjacency (children).
    pub children: Vec<Vec<usize>>,
}

/// Validate nodes, detect cycles, compute topological order and depths.
pub(crate) fn validate(nodes: &[DagNode]) -> MediaResult<ValidateState> {
    let n = nodes.len();

    // --- 1. check duplicate names and build name_map -----------------------
    let mut name_map: HashMap<String, usize> = HashMap::with_capacity(n);
    for (i, node) in nodes.iter().enumerate() {
        if node.name.is_empty() {
            return Err(MediaError::Other(format!("node {} has empty name", i)));
        }
        if name_map.contains_key(&node.name) {
            return Err(MediaError::Other(format!(
                "duplicate node name: '{}'",
                node.name
            )));
        }
        name_map.insert(node.name.clone(), i);
    }

    // --- 2. check dependencies exist and build adjacency lists -------------
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n]; // edge u→v means u must run before v
    let mut indegree: Vec<usize> = vec![0; n];

    for (i, node) in nodes.iter().enumerate() {
        for dep in &node.depends_on {
            let j = name_map.get(dep).ok_or_else(|| {
                MediaError::Other(format!(
                    "node '{}' depends on '{}' which does not exist",
                    node.name, dep
                ))
            })?;
            adj[*j].push(i); // j → i
            indegree[i] += 1;
        }
    }

    // --- 3. Kahn's algorithm for topological sort + cycle detection --------
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut sorted: Vec<usize> = Vec::with_capacity(n);

    for i in 0..n {
        if indegree[i] == 0 {
            queue.push_back(i);
        }
    }

    while let Some(u) = queue.pop_front() {
        sorted.push(u);
        for &v in &adj[u] {
            indegree[v] -= 1;
            if indegree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if sorted.len() != n {
        // Determine cycle participants for a helpful error message.
        let mut cycle_names: Vec<String> = indegree
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg > 0)
            .map(|(i, _)| nodes[i].name.clone())
            .collect();
        cycle_names.sort();
        return Err(MediaError::Other(format!(
            "cycle detected in DAG — remaining nodes with non-zero in-degree: [{}]",
            cycle_names.join(", ")
        )));
    }

    // --- 4. compute topological depth via BFS from sources ----------------
    let mut depth: Vec<usize> = vec![0; n];
    for &u in &sorted {
        for &v in &adj[u] {
            depth[v] = depth[v].max(depth[u] + 1);
        }
    }

    // --- 5. build reverse adjacency (children) for output detection --------
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, adjs) in adj.iter().enumerate() {
        for &v in adjs {
            children[u].push(v);
        }
    }

    Ok(ValidateState {
        sorted,
        name_map,
        depth,
        children,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DagNode;
    use media_compute::EffectKind;

    #[test]
    fn empty_graph() {
        let state = validate(&[]).unwrap();
        assert!(state.sorted.is_empty());
    }

    #[test]
    fn single_node() {
        let nodes = vec![DagNode::new("a", EffectKind::Blur)];
        let state = validate(&nodes).unwrap();
        assert_eq!(state.sorted.len(), 1);
        assert_eq!(state.depth[0], 0);
    }

    #[test]
    fn linear_chain() {
        let nodes = vec![
            DagNode::new("a", EffectKind::Blur),
            DagNode::new("b", EffectKind::Sharpen).depends_on(&["a"]),
            DagNode::new("c", EffectKind::Grayscale).depends_on(&["b"]),
        ];
        let state = validate(&nodes).unwrap();
        // topological: a → b → c
        assert_eq!(state.sorted, vec![0, 1, 2]);
        assert_eq!(state.depth[0], 0);
        assert_eq!(state.depth[1], 1);
        assert_eq!(state.depth[2], 2);
    }

    #[test]
    fn diamond_graph() {
        //      a
        //     / \
        //    b   c
        //     \ /
        //      d
        let nodes = vec![
            DagNode::new("a", EffectKind::Blur),
            DagNode::new("b", EffectKind::Sharpen).depends_on(&["a"]),
            DagNode::new("c", EffectKind::Grayscale).depends_on(&["a"]),
            DagNode::new("d", EffectKind::Invert).depends_on(&["b", "c"]),
        ];
        let state = validate(&nodes).unwrap();
        // a must be first
        assert_eq!(state.sorted[0], 0);
        // d must be last (depends on b and c)
        assert_eq!(state.sorted[3], 3);
        // depths: a=0, b=1, c=1, d=2
        assert_eq!(state.depth[0], 0);
        assert_eq!(state.depth[1], 1);
        assert_eq!(state.depth[2], 1);
        assert_eq!(state.depth[3], 2);
    }

    #[test]
    fn parallel_input() {
        //     a   b
        //      \ /
        //       c
        let nodes = vec![
            DagNode::new("a", EffectKind::Blur),
            DagNode::new("b", EffectKind::Sharpen),
            DagNode::new("c", EffectKind::Grayscale).depends_on(&["a", "b"]),
        ];
        let state = validate(&nodes).unwrap();
        assert_eq!(state.sorted.len(), 3);
        // c must be last (needs both a and b)
        assert_eq!(state.sorted[2], 2);
        // a and b at depth 0, c at depth 1
        assert_eq!(state.depth[0], 0);
        assert_eq!(state.depth[1], 0);
        assert_eq!(state.depth[2], 1);
    }

    #[test]
    fn duplicate_name_detected() {
        let nodes = vec![
            DagNode::new("x", EffectKind::Blur),
            DagNode::new("x", EffectKind::Sharpen),
        ];
        let err = validate(&nodes).unwrap_err();
        assert!(err.to_string().contains("duplicate node name"));
    }

    #[test]
    fn missing_dependency_detected() {
        let nodes = vec![
            DagNode::new("a", EffectKind::Blur),
            DagNode::new("b", EffectKind::Sharpen).depends_on(&["nonexistent"]),
        ];
        let err = validate(&nodes).unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn self_cycle_detected() {
        let nodes = vec![DagNode::new("a", EffectKind::Blur).depends_on(&["a"])];
        let err = validate(&nodes).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn diamond_cycle_detected() {
        // a → b → d
        //  ↖___↘ ↗
        //      c
        let nodes = vec![
            DagNode::new("a", EffectKind::Blur).depends_on(&["d"]),
            DagNode::new("b", EffectKind::Sharpen).depends_on(&["a"]),
            DagNode::new("c", EffectKind::Grayscale).depends_on(&["b"]),
            DagNode::new("d", EffectKind::Invert).depends_on(&["b", "c"]),
        ];
        let err = validate(&nodes).unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn empty_name_detected() {
        let nodes = vec![DagNode::new("", EffectKind::Blur)];
        let err = validate(&nodes).unwrap_err();
        assert!(err.to_string().contains("empty name"));
    }

    #[test]
    fn multi_input_node() {
        let nodes = vec![
            DagNode::new("base", EffectKind::Blur),
            DagNode::new("overlay", EffectKind::Sharpen),
            // simulating a "blend" node that reads two outputs
            DagNode::new("merged", EffectKind::Custom("Blend".into()))
                .depends_on(&["base", "overlay"]),
        ];
        let state = validate(&nodes).unwrap();
        assert_eq!(state.sorted.len(), 3);
        assert_eq!(state.sorted[2], 2); // merged last
        assert_eq!(state.depth[0], 0);
        assert_eq!(state.depth[1], 0);
        assert_eq!(state.depth[2], 1);
    }
}
