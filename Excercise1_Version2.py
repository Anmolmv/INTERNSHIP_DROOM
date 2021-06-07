
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

driver = webdriver.Chrome(executable_path='C:/Users/anmol/OneDrive/Desktop/WebDriver/chromedriver.exe')
driver.get('https://www.g2.com/products/asana/reviews')
results = []
results1 = []
results2 = []
results3 = []
results4 = []
results5 = []
content = driver.page_source
soup = BeautifulSoup(content)
driver.quit()

for a in soup.findAll(attrs='fw-semibold'):
    website = a.find('a')
    if website not in results:
         results.append(website.text)

for b in soup.findAll(attrs='box__heading pb-4th'):
    rating = b.find('h2')
    if rating not in results:
         results1.append(rating.text)

for c in soup.findAll(attrs='filters--product__keyphrase-tag'):
    no_of_reviews = c.find('div')
    if no_of_reviews not in results:
         results2.append(no_of_reviews.text)

for d in soup.findAll(attrs='l5 pb-0'):
    description = d.find('div')
    if description not in results:
         results3.append(description.text)

for e in soup.findAll(attrs='l5 pb-0'):
    seller_details = e.find('div')
    if seller_details not in results:
         results4.append(seller_details.text)

for f in soup.findAll(attrs='l2 pb-half'):
    pricing_details = f.find('h1')
    if pricing_details not in results:
         results5.append(pricing_details.text)

df = pd.DataFrame({'Company': results,'rating': results1, 'no_of_reviews': results2, 'description': results3, 'seller_details': results4, 'pricing_details': results5})
df.to_csv('names3.csv', index=False, encoding='utf-8')
