from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
import csv


prod_info = pd.read_csv('data_scientist_intern_g2_scraper.csv')
industries = prod_info['INDUSTRIES']
names = prod_info['NAME']

f = open("g2scraper.csv", mode='w') # not so sure about the encoding since didn't see the source code
csvwriter = csv.writer(f)

# print(names[0])
opt = Options()
# opt.add_experimental_option('excludeSwitches', ['enable-automation'])
opt.add_argument('--disable-blink-feature=AutomationControlled')


web = Chrome(options=opt)
web.get('https://www.g2.com/')
time.sleep(2)

web.find_element_by_xpath('//*[@id="new_user_consent"]/input[5]').click()
time.sleep(2)

for i in range(len(names)):
    web.find_element_by_xpath('//*[@id="query"]').click()
    web.find_element_by_xpath('//*[@id="query"]').send_keys(names[i], Keys.ENTER)
    time.sleep(5)

# assume no verification code

    # check if there is the product
    product_exist = int(web.find_element_by_xpath('/html/body/div[1]/div/div/div[1]/div/div[3]'
                                                  '/div/div[1]/div[2]/div[2]').text.strip('(').strip(')'))
    if product_exist != 0:

        # get the product information
        number_of_reviews = web.find_element_by_xpath('/html/body/div[1]/div/div/div[1]/div/div[3]/div/div[2]/div[4]/div[1]/div[1]/div/div/a/span[1]').text
        number_of_reviews = int(number_of_reviews.strip('(').strip(')'))

        rating = web.find_element_by_xpath('/html/body/div[1]/div/div/div[1]/div/div[3]/div/div[2]/div[4]/div[1]/div[1]/div/div/a/span[2]/span[1]').text
        rating = int(rating)

        web.find_element_by_xpath('/html/body/div[1]/div/div/div[1]/div/div[3]/div/div[2]/div[4]/div[1]/div[2]/ul/li[1]/a').click()
        time.sleep(2)

        prod_description = web.find_element_by_xpath('//*[@id="leads-sticky-top"]/div/div[1]/div[3]/div[1]/div[1]/div[2]/div/p').text

        prod_url = web.find_element_by_xpath('//*[@id="breadcrumbs"]/li[4]/a').get_attribute('href')

        prod_website = web.find_element_by_xpath('//*[@id="leads-sticky-top"]/div/div[1]/div[3]/div[1]/div[2]/div[2]/div[1]'
                                         '/div[1]/div/div/a').get_attribute('href')

        seller = web.find_element_by_xpath('//*[@id="leads-sticky-top"]/div/div[1]/div[3]/div[1]/div[2]/div[4]/div[2]/div[1]/div[1]/div/div/text()').text

        company_web = web.find_element_by_xpath('//*[@id="leads-sticky-top"]/div/div[1]/div[3]/div[1]/div[2]/div[4]'
                                        '/div[2]/div[1]/div[2]/div/div/a').get_attribute('href')

        csvwriter.writerow([name, number_of_reviews, rating, prod_description, prod_url, prod_website, seller, company_web])

    else:
        print("No information for Company or Product {}".format(names[i]))
        number_of_reviews = 'No information for the company/product'
        rating = 'No information for the company/product'
        prod_description = 'No information for the company/product'
        prod_url = 'No information for the company/product'
        prod_website = 'No information for the company/product'
        seller = 'No information for the company/product'
        company_web ='No information for the company/product'

        csvwriter.writerow([names[i], number_of_reviews, rating, prod_description, prod_url, prod_website, seller, company_web])

print('Over!')
f.close()
web.close()
