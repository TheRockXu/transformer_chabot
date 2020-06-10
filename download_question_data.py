
from bs4 import BeautifulSoup
from urllib import request
from multiprocessing.pool import ThreadPool
import pandas as pd
from tqdm import tqdm

def get_question(link):
    try:
        content = request.urlopen(link)
        soup = BeautifulSoup(content, "html.parser")
        question = soup.find(id='question-header').find('h1').text
        answer = soup.find(class_='answer').find(class_='post-text').find('p').text
        return question, answer
    except Exception as e:
        print(e)

def parse_questions():
    topics = ["ethereum", "bitcoin", "money",
              "law", "workplace", "economics",
              "history", "politics", "pm"]

    results = []

    for topic in topics[5:6]:

        pool = ThreadPool(10)
        file_path = 'data/'+'%s_qlinks.csv'%topic
        print('Processing file - ', file_path)

        df = pd.read_csv(file_path)
        links = [ 'https://%s.stackexchange.com%s'% (topic, v) for v in  df['links'].values]

        res_list = pool.imap_unordered(get_question, links)

        for r in tqdm(res_list):
            if r:
                results.append((r))
    df = pd.DataFrame(results, columns=['questions','answers'])
    df.to_csv('data/data.csv')

if __name__ == '__main__':
    parse_questions()
