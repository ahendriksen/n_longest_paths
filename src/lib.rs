use std::cmp::min;

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

fn argmax(values: &[Option<f32>]) -> usize {
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
    largest_idx
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
        let mut node_pred_edge_idx: Vec<Option<usize>> = vec![None; n];

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
                node_pred_edge_idx[e.to] = Some(i);
            }
        }
        // Compute argmax of node_dist
        let largest_node_idx = argmax(&node_dist);

        // Walk backward from largest node to find all edges in longest path:
        let mut path_length = 0;
        for (edge_idx, _) in iterate_back_from(edges, &node_pred_edge_idx, largest_node_idx) {
            marked[edge_idx] = true;
            path_length += 1;
        }
        num_marked += dbg!(path_length);
        // num_marked += mark_path_from_node(largest_node_idx, &node_pred_edge_idx, &mut marked, edges);
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

        assert_eq!(a, b);
    }
}
