use n_longest_paths::*;
use structopt::StructOpt;
use std::io::Read;
use npy::NpyData;
use npy_derive;
use std::convert::TryInto;

#[derive(Debug, Clone, npy_derive::Serializable)]
pub struct NPEdge {
    pub from: i64,
    pub to: i64,
    pub norm: f32,
}


/// Search for a pattern in a file and display the lines that contain it.
#[derive(StructOpt, Debug)]
struct Cli {
    // The fraction  of edges to find as part of some longest path
    fraction: f64,
    /// The path to the file to read
    #[structopt(parse(from_os_str))]
    input_path: std::path::PathBuf,
    /// The path to the output file
    #[structopt(parse(from_os_str))]
    output_path: std::path::PathBuf,
}


fn main() {
    let args = dbg!(Cli::from_args());

    let mut buf = vec![];
    std::fs::File::open(args.input_path).unwrap()
        .read_to_end(&mut buf).unwrap();

    let data: NpyData<NPEdge> = NpyData::from_bytes(&buf).unwrap();

    let mut edges: Vec<Edge> = Vec::new();

    for e in data {
        let edge = Edge::new(
            e.from.try_into().expect("Got negative 'from' node value"),
            e.to.try_into().expect("Got negative 'to' node value"),
            e.norm);
        // eprintln!("{:?}", &edge);
        edges.push(edge);
    }

    println!("Loaded data");
    println!("Looking for longest paths..");


    let num_to_extract: usize = (args.fraction * (edges.len() as f64)) as usize;
    let in_longest_path = longest_paths_log(&edges, num_to_extract);

}
