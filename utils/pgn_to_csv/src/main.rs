use csv::Writer;
use std::{env, fs::File, io, path::Path};

use pgn_reader::{BufferedReader, Nag, Outcome, RawComment, RawHeader, SanPlus, Visitor};

const DATA_PATH: &str = "/home/florinacho/deepskill/data";

#[derive(Debug)]
struct CSVWriter {
    writer: Writer<File>,
    num_games: u32,
}

const FIELDS: &[&[u8]] = &[
    b"Event",
    b"White",
    b"Black",
    b"Result",
    b"UTCDate",
    b"UTCTime",
    b"WhiteElo",
    b"BlackElo",
    b"Opening",
    b"TimeControl",
    b"Termination",
];

impl CSVWriter {
    fn new<P: AsRef<Path>>(filename: P) -> CSVWriter {
        let mut writer = Writer::from_path(filename).unwrap();
        writer.write_record(FIELDS).unwrap();
        writer.flush().unwrap();
        CSVWriter {
            writer,
            num_games: 0,
        }
    }
}

impl Visitor for CSVWriter {
    type Result = ();

    fn header(&mut self, _key: &[u8], _value: RawHeader<'_>) {
        if FIELDS.contains(&_key) {
            self.writer.write_field(_value.as_bytes()).unwrap();
        }
    }

    fn san(&mut self, _san: SanPlus) {}

    fn nag(&mut self, _nag: Nag) {}

    fn comment(&mut self, _comment: RawComment<'_>) {}

    fn end_variation(&mut self) {}

    fn outcome(&mut self, _outcome: Option<Outcome>) {}

    fn end_game(&mut self) {
        self.writer.write_record(None::<&[u8]>).unwrap();
        self.num_games += 1;
        if self.num_games % 100000 == 0 {
            println!("Game number: {}", self.num_games);
        }
    }
}

fn main() -> Result<(), io::Error> {
    let data_path = Path::new(DATA_PATH);
    let raw_path = data_path.join("raw/");
    let processed_path = data_path.join("processed/");

    for arg in env::args().skip(1) {
        let raw_file = raw_path.join(&arg);
        let file = File::open(&raw_file).expect("Cannot find raw file");

        let uncompressed: Box<dyn io::Read> = Box::new(zstd::Decoder::new(file)?);

        let mut reader = BufferedReader::new(uncompressed);

        let pgn_filename = Path::new(&arg);
        let csv_filename = pgn_filename.with_extension("").with_extension("csv");
        let csv_file = processed_path.join(csv_filename);

        let mut writer = CSVWriter::new(csv_file);
        reader.read_all(&mut writer)?;
        writer.writer.flush().unwrap();
    }

    Ok(())
}
