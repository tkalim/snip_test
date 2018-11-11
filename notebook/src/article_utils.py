import pandas as pd
from newspaper import Article


def get_article_data(url):
    """Downloads article and rerturns relevant data
    """
    article = Article(url)
    article.download()
    article.parse()
    author = " & ".join(list(article.authors))
    text = article.text
    title = article.title
    # article_dict = {"title": [title], "author": [author], "text": [text]}
    # article_dataframe = pd.DataFrame(list(article_dict.items()), columns=["title", "author", "text"])
    # article_dataframe = pd.DataFrame([article_dict])
    # article_dataframe = pd.DataFrame(article_dict)
    # return article_dataframe
    return title, author, text


def preprocess_article_data(article_dataframe, vectorizer, transformer):
    """preprocess the article dataframe
    returns a sparse matrix
    """
    article_dataframe["total"] = article_dataframe["title"] + " " + article_dataframe["author"] + article_dataframe["text"]
    vector = vectorizer.transform(article_dataframe)
    sparse_vector = transformer.transform(vector)
    return sparse_vector

def rate_articles(url_list, model, vectorizer, transformer):

    reliability_scores = []
    titles, authors, texts = [], [], []
    for url in url_list:
        # article_dataframe = get_article_data(url)
        title, author, text = get_article_data(url)
        titles.append(title)
        authors.append(author)
        texts.append(text)
        # title, author = article_dataframe["title"][0], article_dataframe["author"][0]
        # article_vector = preprocess_article_data(article_dataframe, vectorizer, transformer)
        # reliability_score = model.predict_proba(article_vector)
        # print(f"{title} by {author}:\t{reliability_score}")
        # reliability_scores.append(reliability_score)
    articles_dict = {"title": titles, "author": authors, "text": texts}
    articles_dataframe = pd.DataFrame(articles_dict)
    print(articles_dataframe.head())
    articles_vector = preprocess_article_data(articles_dataframe, vectorizer, transformer)
    reliability_scores = model.predict_proba(articles_vector)

    return reliability_scores

