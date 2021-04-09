use std::cmp::min;

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub len: f32,
}

impl Edge {
    pub fn new(from:usize, to:usize, len: f32) -> Edge{
        Edge{ from, to, len}
    }
}

/// Run longest_paths in steps and clean up used edges in between.
pub fn longest_paths_log(edges: &[Edge], num_to_extract: usize) -> Vec<bool> {
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
        let extracted_edges = longest_paths(&shortened_edges, num_to_extract);
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


pub fn longest_paths(edges: &[Edge], extract_num: usize) -> Vec<bool> {
    for e in edges.iter(){
        assert!(e.from < e.to);
    }
    assert!(! edges.is_empty());

    // Assert that edges are sorted
    for i in 1..edges.len(){
        let e0 = &edges[i - 1];
        let e1 = &edges[i];

        let e0 = (e0.from, e0.to);
        let e1 = (e1.from, e1.to);

        assert!(e0 <= e1);
    }

    let extract_num = min(extract_num, edges.len());
    let mut num_extracted: usize = 0;
    let mut in_longest_path = vec![false; edges.len()];

    // n - 1 is the index of the largest node
    let n: usize = edges.iter().map(|e| e.to).max().unwrap() + 1;

    while num_extracted < extract_num {
        let mut node_dist: Vec<Option<f32>> = vec![None; n];
        let mut node_pred_edge_idx: Vec<Option<usize>> = vec![None; n];

        // Compute node distances
        for (i, e) in edges.iter().enumerate() {
            if in_longest_path[i] {
                continue;
            }
            let new_dist = node_dist[e.from].unwrap_or(0.0) + e.len;
            let old_dist = node_dist[e.to];

            let replace = match old_dist {
                Some(old_dist) => old_dist <= new_dist ,
                None => true,
            };
            if replace
            {
                node_dist[e.to] = Some(new_dist);
                node_pred_edge_idx[e.to] = Some(i);
            }
        }
        // Compute argmax of node_dist
        let mut largest_dist = f32::NEG_INFINITY;
        let mut largest_node: usize = 0;
        for (i, dist) in node_dist.iter().enumerate(){
            if let Some(d) = dist {
                if &largest_dist <= d {
                    largest_dist = *d;
                    largest_node = i;
                }
            }
        }

        // Walk backward from largest node to find all edges in longest path:
        let mut cur_node = largest_node;
        loop {
            if let Some(pred_edge_idx) = node_pred_edge_idx[cur_node] {
                let pred_edge = &edges[pred_edge_idx];
                // Indicate that edge was part of longest_path
                in_longest_path[pred_edge_idx] = true;
                num_extracted += 1;

                // Walk path:
                cur_node = pred_edge.from;
            } else {
                break
            }
        }
    }

    in_longest_path
}


mod tests {
    #[test]
    fn longest_paths_works() {
        use super::*;
        let edges = vec![
            Edge::new(0, 1, 0.1),
            Edge::new(0, 1, 1.1),
            Edge::new(0, 1, 0.5),
            Edge::new(0, 2, 5.0),
            Edge::new(1, 2, 1.0),
        ];
        assert_eq!(longest_paths(&edges, 1), vec![false, false, false, true, false]);
        assert_eq!(longest_paths(&edges, 3), vec![false, true, false, true, true]);
        assert_eq!(longest_paths(&edges, 4), vec![false, true, true, true, true]);
    }

    #[test]
    fn longest_path_consistent() {
        use super::*;
        let n = 256;
        let mut edges: Vec<Edge> = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in (i+1)..n {
                edges.push(Edge::new(i, j, (i + j) as f32));
            }
        }

        let a = longest_paths(&edges, 20_000);
        let b = longest_paths_log(&edges, 20_000);

        assert_eq!(a, b);
    }
}
