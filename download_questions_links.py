"""
Download Quesstion Links from stackexchange
"""
from bs4 import BeautifulSoup
from urllib import request
from tqdm import tqdm
import pandas as pd
import time

def get_question_links(topic, pages):
    """
    Parse all the raw links to question pages
    :return:
    """
    links = []
    for i in tqdm(range(pages)):
        try:
            money_url = "https://%s.stackexchange.com/questions?tab=votes&pagesize=50&page=%d"%(topic, i)
            content = request.urlopen(money_url)
            soup = BeautifulSoup(content, "html.parser")
            qsummaries = soup.find_all(class_='summary')
            links.extend([l.find('h3').find('a')['href'] for l in qsummaries])
        except Exception as e:
            print(e)
            time.sleep(10)
    df = pd.DataFrame(links, columns=['links'])
    df.to_csv('data/%s_qlinks.csv' % topic)
    print("%s Data Collected"%topic)
if __name__ == '__main__':
    topics = { "ethereum":100,"bitcoin":100, "money":300,
               "law":100,"workplace":100,"economics":100,
               "history":100, "politics":100, "pm":100}
    for topic, pages in topics.items():
        get_question_links(topic, pages)
        # break
