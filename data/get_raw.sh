if [[ $# -ne 1 ]]; then
    echo "Must input year and month in yyyy-mm format"
    exit 1
fi

date=$1

if [[ ! $date =~ ^20[0-2][0-9]-[0-1][0-9]$ ]]; then
    echo "Given input must be in yyyy-mm format"
    exit 1
fi

url="https://database.lichess.org/standard/lichess_db_standard_rated_$date.pgn.zst"
filename="lichess_db_standard_rated_$date.pgn"

data_dir=$(dirname "$0")
raw_dir="$data_dir/raw"

file="$raw_dir/$filename"

if [[ ! -f "$file.zst" ]]; then
    wget -P "$raw_dir" "$url"
else
    echo "Data already downloaded"
fi

if [[ ! -f $file ]]; then
    unzstd "$raw_dir/$filename.zst"
else
    echo "Data already uncompressed"
fi
