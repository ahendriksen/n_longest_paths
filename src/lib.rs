use std::cmp::min;
use std::collections::{HashSet, BinaryHeap};
use std::cmp::Reverse;
use pbr::ProgressBar;

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub len: f32,
}

impl Edge {
    pub fn new(from:usize, to:usize, len: f32) -> Edge{
        Edge{from, to, len}
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NormGroup {
    Additive,
    Multiplicative
}

use NormGroup::*;

/// Run longest_paths in steps and clean up used edges in between.
pub fn mark_longest_paths_stepwise(edges: &[Edge], num_to_extract: usize, norm_group: NormGroup) -> Vec<bool> {
    const MIN_STEP_SIZE:usize = 1000;

    let num_to_extract = min(num_to_extract, edges.len());

    let mut shortened_edges = Vec::from(edges);

    let mut in_shortest_path = vec![false; edges.len()];
    let total_num_extract = num_to_extract;
    let mut num_extracted = 0;

    while num_extracted < total_num_extract {
        let mut num_to_extract = total_num_extract - num_extracted;
        if MIN_STEP_SIZE < num_to_extract {
            num_to_extract = num_to_extract / 2;
        }
        let extracted_edges: Vec<bool> = mark_longest_paths(&shortened_edges, num_to_extract, norm_group);

        num_extracted += extracted_edges.iter().map(|&b| if b {1} else {0}).sum::<usize>();

        // Update in_shortest_path
        let mut i = 0;
        let mut j = 0;
        while i < in_shortest_path.len() {
            if in_shortest_path[i] {
                i += 1;
            } else if extracted_edges[j]  {
                in_shortest_path[i] = true;
                j += 1;
                i += 1;
            } else {
                i += 1;
                j += 1;
            }
        }

        // Update shortened_edges:
        shortened_edges = shortened_edges
            .iter()
            .zip(extracted_edges.iter())
            .filter_map(|(e, &extracted)| if extracted{ None } else {Some(e.clone())})
            .collect();
    }
    in_shortest_path
}


fn is_forward_pointing(edges: &[Edge]) -> bool {
    for e in edges.iter(){
        if e.to <= e.from {
            return false;
        }
    }
    true
}

fn is_sorted(edges: &[Edge]) -> bool {
    for i in 1..edges.len(){
        let e0 = &edges[i - 1];
        let e1 = &edges[i];

        let e0 = (e0.from, e0.to);
        let e1 = (e1.from, e1.to);

        if e1 < e0 {
            return false;
        }
    }
    true
}

fn argmax(values: &[Option<f32>]) -> (usize, f32) {
    let mut largest_val = f32::NEG_INFINITY;
    let mut largest_idx: usize = 0;
    for (i, val) in values.iter().enumerate(){
        if let Some(val) = val {
            if &largest_val <= val {
                largest_val = *val;
                largest_idx = i;
            }
        }
    }
    (largest_idx, largest_val)
}


struct BackwardEdgeIterator<'a>{
    edges: &'a[Edge],
    incoming_edge_idx: &'a[Option<usize>],
    node_idx: usize,
}

impl<'a> Iterator for BackwardEdgeIterator<'a> {
    type Item = (usize, &'a Edge);

    // The method that generates each item
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(incoming_edge_idx) = self.incoming_edge_idx[self.node_idx] {
            let incoming_edge = &self.edges[incoming_edge_idx];
            self.node_idx = incoming_edge.from;
            Some((incoming_edge_idx, incoming_edge))
        } else {
            None
        }
    }
}

fn iterate_back_from<'a>(
    edges: &'a[Edge],
    incoming_edge_idx: &'a[Option<usize>],
    node_idx: usize
) -> BackwardEdgeIterator<'a> {
    BackwardEdgeIterator{
        edges,
        incoming_edge_idx,
        node_idx,
    }
}

fn compute_node_distances(edges: &[Edge], norm_group: NormGroup) -> (Vec<Option<f32>>, Vec<Option<usize>>) {
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;

    let mut node_dist: Vec<Option<f32>> = vec![None; n];
    let mut node_incoming_edge_idx: Vec<Option<usize>> = vec![None; n];

    // Compute node distances
    for (i, e) in edges.iter().enumerate() {
        let new_dist = compute_dist(node_dist[e.from], &e, norm_group);
        let old_dist = node_dist[e.to];

        if do_replace(old_dist, new_dist, norm_group)
        {
            node_dist[e.to] = Some(new_dist);
            node_incoming_edge_idx[e.to] = Some(i);
        }
    }
    (node_dist, node_incoming_edge_idx)
}


/// Mark//  edges that are in a longest_path. Continue until at least `num_to_mark`
/// edges are marked. Once an edge has been marked, it is ignored for further
/// computations of longest paths.
pub fn mark_longest_paths_faster(edges: &[Edge], num_to_mark: usize, norm_group: NormGroup) -> Vec<bool> {
    assert!(! edges.is_empty());
    assert!(is_forward_pointing(edges));
    assert!(is_sorted(edges));

    let num_to_mark = min(num_to_mark, edges.len());
    let mut num_marked: usize = 0;
    let mut marked = vec![false; edges.len()];

    // For each node, prepare a hashset containing all its incoming edges;
    // Also, prepare a hashset containing all its outgoing edges.
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;
    let mut node_all_incoming_edge_idxs: Vec<HashSet<usize>> = vec![HashSet::with_capacity(edges.len() / n); n];
    let mut node_all_outgoing_edge_idxs: Vec<HashSet<usize>> = vec![HashSet::with_capacity(edges.len() / n); n];

    let mut pb = ProgressBar::new(edges.len() as u64);
    for (i, e) in edges.iter().enumerate() {
        if let Some(node_incoming_edge_set) = node_all_incoming_edge_idxs.get_mut(e.to) {
            node_incoming_edge_set.insert(i);
        }
        if let Some(node_outgoing_edge_set) = node_all_outgoing_edge_idxs.get_mut(e.from) {
            node_outgoing_edge_set.insert(i);
        }
        pb.tick();
    }
    pb.finish_print("Finished preparing.");

    let (node_dist, node_incoming_edge_idx) = compute_node_distances(edges, norm_group);
    let mut node_dist: Vec<Option<f32>> = node_dist;
    let mut node_incoming_edge_idx: Vec<Option<usize>> = node_incoming_edge_idx;

    // Create progress bar
    let mut pb = ProgressBar::new(num_to_mark as u64);

    while num_marked < num_to_mark {
        pb.set(num_marked as u64);
        // Compute argmax of node_dist
        let (largest_node_idx, _) = argmax(&node_dist);

        // Walk backward from largest node to find all edges in longest path:
        let mut path_marked_edges: Vec<usize> = Vec::with_capacity(n);
        for (edge_idx, _) in iterate_back_from(edges, &node_incoming_edge_idx, largest_node_idx) {
            path_marked_edges.push(edge_idx);
        }

        let num_marked_edges_in_path = path_marked_edges.len();
        num_marked += num_marked_edges_in_path;

        // Walk forward through path and
        // - mark edges along the path
        // - remove edges from hashsets
        // - mark nodes along path for invalidation
        let mut nodes_to_recalculate: BinaryHeap<Reverse<usize>> = BinaryHeap::with_capacity(n);
        while let Some(edge_idx) = path_marked_edges.pop() {
            // Mark edge
            marked[edge_idx] = true;
            let edge = &edges[edge_idx];
            // print!("({0} -- {1} : {2:0.2}) ", edge.from, edge.to, edge.len);

            // Remove edge as incoming edge from destination node:
            let dest_node_idx = edge.to;
            if let Some(node_incoming_edge_set) = node_all_incoming_edge_idxs.get_mut(dest_node_idx) {
                node_incoming_edge_set.remove(&edge_idx);
            }

            // Remove edge as outgoing edge from source node:
            let src_node_idx = edge.from;
            if let Some(node_outgoing_edge_set) = node_all_outgoing_edge_idxs.get_mut(src_node_idx) {
                node_outgoing_edge_set.remove(&edge_idx);
            }

            // Recalculate destination node's distance. This will be updated later.
            nodes_to_recalculate.push(Reverse(dest_node_idx));
        }

        // Recalculate downstream nodes
        while let Some(Reverse(dest_node_idx)) = nodes_to_recalculate.pop() {
            // We are uing a min-heap, so at any point in time, node_idx is the
            // smallest node that is in invalid state. Hence, after invalidating
            // the current computed distance, we can recalculate it on the basis
            // of the *valid* incoming edge and node values.
            node_dist[dest_node_idx] = None;
            node_incoming_edge_idx[dest_node_idx] = None;

            // Recalculate node_distance
            for &edge_idx in (&node_all_incoming_edge_idxs[dest_node_idx]).iter() {
                let edge = &edges[edge_idx];
                assert!(edge.to == dest_node_idx);
                let src_node_idx = edge.from;
                let current_dist = node_dist[dest_node_idx];
                let new_dist = compute_dist(node_dist[src_node_idx], &edge, norm_group);

                if do_replace(current_dist, new_dist, norm_group) {
                    node_dist[dest_node_idx] = Some(new_dist);
                    node_incoming_edge_idx[dest_node_idx] = Some(edge_idx);
                }
            }

            // Mark downstream nodes from recalculation
            for &edge_idx in (&node_all_outgoing_edge_idxs[dest_node_idx]).iter() {
                // Recalculate downstream node if it is connected to current node.
                let downstream_node_idx = edges[edge_idx].to;
                if let Some(incoming_edge) = node_incoming_edge_idx[downstream_node_idx] {
                    if edges[incoming_edge].from == dest_node_idx {
                        nodes_to_recalculate.push(Reverse(downstream_node_idx));
                    }
                }
            }
        }
        if num_marked_edges_in_path == 0 {
            // All edges are 'negative' (either < 0 or < 1) and thus paths
            // cannot be formed anymore. Proceed to individual pruning..
            break;
        }
    }

    // Continue with individual pruning
    pb.message("Continuing with individual pruning!");
    let mut norm_idx_pairs: Vec<_> = edges.iter()
         .enumerate()
         .filter_map(
             |(i, edge)| if marked[i] {
                 None
             } else {
                 Some((edge.len, i))
             }
         )
        .collect();

    norm_idx_pairs.sort_by(|(n1, _), (n2, _)| n1.partial_cmp(n2).unwrap());

    for (_, i) in norm_idx_pairs.iter() {
        pb.set(num_marked as u64);
        if num_to_mark <= num_marked {
            break;
        }
        marked[*i] = true;
        num_marked += 1;
    }
    pb.finish_print("done");
    marked
}

fn compute_dist(src_node_dist: Option<f32>, edge: &Edge, norm_group: NormGroup) -> f32 {
    match norm_group {
        Additive => src_node_dist.unwrap_or(0.0) + edge.len,
        Multiplicative => src_node_dist.unwrap_or(1.0) * edge.len,
    }
}

fn do_replace(current_dist: Option<f32>, proposed_dist: f32, norm_group: NormGroup) -> bool {
    match current_dist {
        None => match norm_group {
            Additive => 0.0 <= proposed_dist,
            Multiplicative => 1.0 <= proposed_dist,
        }
        Some(current_dist) => current_dist <= proposed_dist,
    }
}

/// Mark edges that are in a longest_path. Continue until at least `num_to_mark`
/// edges are marked. Once an edge has been marked, it is ignored for further
/// computations of longest paths.
pub fn mark_longest_paths(edges: &[Edge], num_to_mark: usize, norm_group: NormGroup) -> Vec<bool> {
    assert!(! edges.is_empty());
    assert!(is_forward_pointing(edges));
    assert!(is_sorted(edges));

    let num_to_mark = min(num_to_mark, edges.len());
    let mut num_marked: usize = 0;
    let mut marked = vec![false; edges.len()];

    // n - 1 is the index of the largest node
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;

    while num_marked < num_to_mark {
        let mut node_dist: Vec<Option<f32>> = vec![None; n];
        let mut node_incoming_edge_idx: Vec<Option<usize>> = vec![None; n];

        // Compute node distances
        for (i, e) in edges.iter().enumerate() {
            if marked[i] {
                continue;
            }

            let new_dist = match norm_group {
                Additive => node_dist[e.from].unwrap_or(0.0) + e.len,
                Multiplicative => node_dist[e.from].unwrap_or(1.0) * e.len,
            };

            let old_dist = node_dist[e.to];

            // Replace node distance if it is None or if new_dist is larger.
            let replace = match old_dist {
                None => true,
                Some(old_dist) => old_dist <= new_dist ,
            };
            if replace
            {
                node_dist[e.to] = Some(new_dist);
                node_incoming_edge_idx[e.to] = Some(i);
            }
        }
        // Compute argmax of node_dist
        let (largest_node_idx, _longest_path_len) = argmax(&node_dist);

        // Walk backward from largest node to find all edges in longest path:
        let mut path_length = 0;
        for (edge_idx, edge) in iterate_back_from(edges, &node_incoming_edge_idx, largest_node_idx) {
            marked[edge_idx] = true;
            dbg!(edge);
            path_length += 1;
        }
        num_marked += dbg!(path_length);
    }

    marked
}


mod tests {
    #[test]
    fn longest_paths_additive_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 0.1),
            Edge::new(0, 1, 1.1),
            Edge::new(0, 1, 0.5),
            Edge::new(0, 2, 5.0),
            Edge::new(1, 2, 1.0),
        ];
        assert_eq!(mark_longest_paths(&edges, 1, Additive), vec![false, false, false, true, false]);
        assert_eq!(mark_longest_paths(&edges, 3, Additive), vec![false, true, false, true, true]);
        assert_eq!(mark_longest_paths(&edges, 4, Additive), vec![false, true, true, true, true]);
    }


    #[test]
    fn longest_paths_mult_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 1.0),
            Edge::new(0, 1, 2.0),
            Edge::new(0, 1, 3.0),
            Edge::new(0, 2, 7.0),
            Edge::new(1, 2, 3.0),
        ];
        assert_eq!(mark_longest_paths(&edges, 1, Multiplicative), vec![false, false, true, false, true]);
        assert_eq!(mark_longest_paths(&edges, 2, Multiplicative), vec![false, false, true, false, true]);
        assert_eq!(mark_longest_paths(&edges, 3, Multiplicative), vec![false, false, true, true, true]);
        assert_eq!(mark_longest_paths(&edges, 4, Multiplicative), vec![false, true, true, true, true]);
        assert_eq!(mark_longest_paths(&edges, 5, Multiplicative), vec![true, true, true, true, true]);

        assert_eq!(mark_longest_paths(&edges, 1, Additive), vec![false, false, false, true, false]);
        assert_eq!(mark_longest_paths(&edges, 2, Additive), vec![false, false, true, true, true]);
        assert_eq!(mark_longest_paths(&edges, 3, Additive), vec![false, false, true, true, true]);
        assert_eq!(mark_longest_paths(&edges, 4, Additive), vec![false, true, true, true, true]);
        assert_eq!(mark_longest_paths(&edges, 5, Additive), vec![true, true, true, true, true]);
    }

#[test]
    fn longest_paths_faster_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 1.0),
            Edge::new(0, 1, 2.0),
            Edge::new(0, 1, 3.0),
            Edge::new(0, 2, 7.0),
            Edge::new(1, 2, 3.0),
        ];
        assert_eq!(mark_longest_paths_faster(&edges, 1, Multiplicative), vec![false, false, true, false, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 2, Multiplicative), vec![false, false, true, false, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 3, Multiplicative), vec![false, false, true, true, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 4, Multiplicative), vec![false, true, true, true, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 5, Multiplicative), vec![true, true, true, true, true]);

        assert_eq!(mark_longest_paths_faster(&edges, 1, Additive), vec![false, false, false, true, false]);
        assert_eq!(mark_longest_paths_faster(&edges, 2, Additive), vec![false, false, true, true, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 3, Additive), vec![false, false, true, true, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 4, Additive), vec![false, true, true, true, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 5, Additive), vec![true, true, true, true, true]);

        // Check that long chains are correctly handled:
        // Full chain is worth 5 * 5 / 2 = 12.5
        // sub-chains are worth as most 5.
        let edges = vec![
            Edge::new(0, 1, 5.0),
            Edge::new(1, 2, 0.5),
            Edge::new(2, 3, 5.0),
        ];
        assert_eq!(mark_longest_paths_faster(&edges, 1, Multiplicative), vec![true, true, true]);

        // Check that node distances are also forward propagated
        let edges = vec![
            Edge::new(0, 1, 2.0),
            Edge::new(1, 2, 0.5),
            Edge::new(2, 3, 1.5),
        ];
        assert_eq!(mark_longest_paths_faster(&edges, 1, Multiplicative), vec![true, false, false]);
        println!("\n\nTEST that FAILS");
        println!("------------------------------------------------------------");
        assert_eq!(mark_longest_paths_faster(&edges, 2, Multiplicative), vec![true, false, true]);
        assert_eq!(mark_longest_paths_faster(&edges, 3, Multiplicative), vec![true, true, true]);

    }

    #[test]
    fn longest_path_consistent() {
        use super::*;
        let n = 64;
        let mut edges: Vec<Edge> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in (i+1)..n {
                edges.push(Edge::new(i, j, (i + j) as f32));
            }
        }

        let a = mark_longest_paths(&edges, 20_000, Additive);
        let b = mark_longest_paths_stepwise(&edges, 20_000, Additive);
        let c = mark_longest_paths_faster(&edges, 20_000, Additive);
        assert_eq!(a, b);
        assert_eq!(a, c);
    }
}
