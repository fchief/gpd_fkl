import os
import re
import csv
import numpy as np
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk import PorterStemmer
import zipfile

# Global data for processing use
dataset = {'Sales_Transactions_Dataset_Weekly.csv': 'https://www.kaggle.com/crawford/weekly-sales-transactions/downloads/Sales_Transactions_Dataset_Weekly.csv', 
            'data_job_posts.zip': 'https://www.kaggle.com/madhab/jobposts/downloads/data%20job%20posts.csv'}
fields = ["TITLE:", "TERM:",  "DURATION:", "LOCATION:", 
    "JOB DESCRIPTION:", "RESPONSIBILITIES:", "QUALIFICATIONS:",
    "SALARY:", "REMUNERATION:", "DEADLINE:", "COMPANY:"]
clean_fields = ["TITLE:",  "DURATION:", "LOCATION:", 
    "JOB DESCRIPTION:", "RESPONSIBILITIES:", "QUALIFICATIONS:",
    "SALARY:", "DEADLINE:", "COMPANY:"]


# ------------------------------------------------------------------------------------------
# This function takes string input and remove stop words and singularize the words.
def str_swr_sgl(str_input):

# ------------------------------------------------------------------------------------------
# This function takes a dataframe as input and outputs total number of job posts
# by month over the years from 2015-2004. The column is numeric month and the row is year.
def jobpost_month(df):
    import calendar
    months = df["Month"].unique().tolist()
    years = df["Year"].unique().tolist()
    results = pd.DataFrame(index=years, columns=months)
    for y in years:
        df_y = df["Year"] == y
        results.loc[y] = df[df_y]["Month"].value_counts()
    return(results)

# ------------------------------------------------------------------------------------------
# This function takes a dataframe as input and outputs the company name with the most number
# of job posts in the past two years since 2015.
def jobpost_company(df):
    year2013 = df["Year"] >= 2013
    jobs_year2013 = df[year2013]["Company"].value_counts().index.tolist()
    print("Company {0} has most job post ads from year 2013-2015.".format(jobs_year2013[0]))
    return(jobs_year2013[0])

# ------------------------------------------------------------------------------------------
# This function takes a string input containing job post details and search for job post 
# details shown in clean_fields list. List containing the results is returned.
def jobpost_scraper(jobpost_str):
    regex = r"[A-Z / A-Z]{4,}\:"
    job_post = []
    matches = re.finditer(regex, jobpost_str)
    match_items = []
    match_headers = []
    for match in matches:
        h = jobpost_str[match.start():match.end()].strip()
        for f in fields:
            if f in h:
                if f == "SALARY:":
                    h = "SALARY:"
                if f == "TERM:":
                    h = "DURATION:"
                else:
                    h = f
        match_headers.append(h)
        match_items.append((match.start(), match.end()))
    nn = len(clean_fields)
    mm = len(match_headers)
    df_ls = []
    for cf in clean_fields:
        if cf in match_headers:
            ii = match_headers.index(cf)
            # print(match_headers[ii] + "----------------------------")
            if ii < mm - 1:
                idx_s = match_items[ii][1]
                idx_e = match_items[ii+1][0]
                res = jobpost_str[idx_s:idx_e].strip()
            else:
                idx_s = match_items[ii][1]
                res = jobpost_str[idx_s:].strip()
            idx_ = res.find("--")
            if idx_ != -1:
                res = res[0:idx_].strip()
            res = res.replace("\r\n", "")
            # result.at[i, match_headers[ii]] = res
            df_ls.append(res)
        else:
            # result.at[i, cf] = "NULL"
            df_ls.append("NULL")
    return df_ls
        
# This function is a caller function to invoke jobpost related eda functions
def jobpost_eda():
    zf = zipfile.ZipFile("data_job_posts.zip")
    df = pd.read_csv(zf.open("data job posts.csv"))

    jobpost_company(df)
    jobpost_month(df)
    
    n = len(df["jobpost"])
    # regex = r"[A-Z / A-Z]{4,}\:"
    result = pd.DataFrame(columns = clean_fields)
    for i in range(0,n):
        temp = df["jobpost"][i]
        result.loc[i] = jobpost_scraper(temp)
    print("Total {0} job posts scraped.".format(n))
    print(result[:6])
    #     job_post = []
    #     matches = re.finditer(regex, temp)
    #     match_items = []
    #     match_headers = []
    #     for match in matches:
    #         h = temp[match.start():match.end()].strip()
    #         for f in fields:
    #             if f in h:
    #                 if f == "SALARY:":
    #                     h = "SALARY:"
    #                 if f == "TERM:":
    #                     h = "DURATION:"
    #                 else:
    #                     h = f
    #         match_headers.append(h)
    #         match_items.append((match.start(), match.end()))
    #     nn = len(clean_fields)
    #     mm = len(match_headers)
    #     df_ls = []
    #     for cf in clean_fields:
    #         if cf in match_headers:
    #             ii = match_headers.index(cf)
    #             # print(match_headers[ii] + "----------------------------")
    #             if ii < mm - 1:
    #                 idx_s = match_items[ii][1]
    #                 idx_e = match_items[ii+1][0]
    #                 res = temp[idx_s:idx_e].strip()
    #             else:
    #                 idx_s = match_items[ii][1]
    #                 res = temp[idx_s:].strip()
    #             idx_ = res.find("--")
    #             if idx_ != -1:
    #                 res = res[0:idx_].strip()
    #             res = res.replace("\r\n", "")
    #             # result.at[i, match_headers[ii]] = res
    #             df_ls.append(res)
    #         else:
    #             # result.at[i, cf] = "NULL"
    #             df_ls.append("NULL")
    #     result.loc[i] = df_ls
    # print("Total {0} job posts scraped.".format(n))
    # print(result[:6])

    

def is_outlier(value, qtr25, qtr75):
    iqr = qtr75 - qtr25
    lower = qtr25 - 1.5*iqr
    upper = qtr75 + 1.5*iqr
    return (value < lower) or (value > upper)

def audit_sum(input):
    total = []
    [total.append(int(i)) for i in input if i.isdigit()]
    return sum(total)

def sale_analysis():
    filename = "Sales_Transactions_Dataset_Weekly.csv"
    results = {}
    with open(filename, "r") as input_f:
        reader = csv.reader(input_f, delimiter=",")
        header = next(reader)
        biweekly = np.arange(1,53,2)
        biweekly = biweekly.tolist()
        total_sales_by_prod = {}
        outliers = {}
        total_biweekly_sales = []
        for r in reader:
            int_r = []
            outlier_idx = []
            [int_r.append(int(r[i])) for i in range(1,54)]
            qtr25, qtr75 = np.percentile(int_r, [25,75])
            for i in range(1,53):
                if is_outlier(int(r[i]), qtr25, qtr75):
                    outlier_idx.append(header[i])
            outliers[r[0]] = outlier_idx
            biweekly_sales = [r[0]]
            for b in biweekly:
                biweekly_sales.append(audit_sum(r[b:b+2]))
            total_biweekly_sales.append(biweekly_sales)
            total_sales_by_prod[r[0]] = audit_sum(r[1:53])
        results["Best Performing"] = max(total_sales_by_prod, key=total_sales_by_prod.get)   
        results["Outliers"] = outliers          
        df = pd.DataFrame(total_biweekly_sales)
        print(total_sales_by_prod["P409"])
        # print(min(df[20]))



def sum_exc(input):
    val = []
    n = len(input)

    # Deep copy approach
    # for i in range(0,n):
    #     temp = input[:]
    #     temp.pop(i)
    #     val.append(sum(temp))

    # Pythonic way using list slicing
    [val.append(sum(input[:i] + input[i+1:])) for i in range(0,n)]
    return val

def get_kaggle_dataset(key, link):
    # You need to input your kaggle accounts for authentication
    login_url = "https://www.kaggle.com/account/login?isModal=true"
    payload =  {"username" : "fchief", "password" : "earth169", "__RequestVerificationToken" : ""}
    r = requests.session()
    result = r.get(login_url)
    payload["__RequestVerificationToken"] = r.cookies.get_dict()["__RequestVerificationToken"]
    result = r.post(login_url, data=payload, headers = dict(referer=login_url) )
    # print(result.content)
    result = r.get(
	link, 
	headers = dict(referer = link)
    )

    with open(key, 'wb') as output_f:
        for byte in result.iter_content(chunk_size=524288):
            if byte:
                output_f.write(byte)

    return r



if __name__ == '__main__':
    # This part checks if dataset exists, otherwise download from Kaggle. Tested under Ubuntu 16.04 OS
    cur_path = "."
    cur_dir_files = os.listdir(cur_path)
    [get_kaggle_dataset(key, dataset[key]) for key in dataset.keys() if key not in cur_dir_files]

    # Question 1
    input = [1,2,3,4] # You may change the list entries here. No numeric checking
    val = sum_exc(input)
    print("Q1 - Input list: {0} Index exclusion sum: {1}".format(input, val))
    sale_analysis()
    jobpost_eda()