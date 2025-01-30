set -x
ariac2 https://business.yelp.com/external-assets/files/Yelp-JSON.zip
unzip Yelp-JSON.zip
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Industrial_and_Scientific.jsonl.gz
gzip -d Industrial_and_Scientific.jsonl.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz
gzip -d meta_Industrial_and_Scientific.jsonl.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Industrial_and_Scientific.csv.gz
gzip -d Industrial_and_Scientific.csv.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Musical_Instruments.jsonl.gz
gzip -d Musical_Instruments.jsonl.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Musical_Instruments.jsonl.gz
gzip -d meta_Musical_Instruments.jsonl.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv.gz
gzip -d Musical_Instruments.csv.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
gzip -d Video_Games.jsonl.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz
gzip -d meta_Video_Games.jsonl.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz
gzip -d Video_Games.csv.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz
gzip -d goodreads_reviews_children.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_children.json.gz
gzip -d goodreads_books_children.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz
gzip -d goodreads_reviews_comics_graphic.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz
gzip -d goodreads_books_comics_graphic.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz
gzip -d goodreads_reviews_poetry.json.gz
wget https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_poetry.json.gz
gzip -d goodreads_books_poetry.json.gz
