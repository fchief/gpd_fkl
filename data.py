import os
import re
import csv
import numpy as np
import pandas as pd
import requests
import zipfile
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer, pos_tag
from pattern.en import singularize
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------------------------------------
# Download files if not available
# import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

# Global data for processing use
dataset = {'Sales_Transactions_Dataset_Weekly.csv': 'https://www.kaggle.com/crawford/weekly-sales-transactions/downloads/Sales_Transactions_Dataset_Weekly.csv', 
            'data_job_posts.zip': 'https://www.kaggle.com/madhab/jobposts/downloads/data%20job%20posts.csv',
            "test.csv":"https://www.kaggle.com/rishisankineni/text-similarity/downloads/test.csv"}
fields = ["TITLE:", "TERM:",  "DURATION:", "LOCATION:", 
    "JOB DESCRIPTION:", "RESPONSIBILITIES:", "QUALIFICATIONS:",
    "SALARY:", "REMUNERATION:", "DEADLINE:", "COMPANY:"]
clean_fields = ["TITLE:",  "DURATION:", "LOCATION:", 
    "JOB DESCRIPTION:", "RESPONSIBILITIES:", "QUALIFICATIONS:",
    "SALARY:", "DEADLINE:", "COMPANY:"]

# ------------------------------------------------------------------------------------------
# To parallelize this function one can split dataframe into partitions and call this function
# concurrently. The python library commonly used is multiprocess. 
def text_sim(df):
    input = [df["description_x"], df["description_y"]]
    # Cosine similarity calculates the angle of two vectors of words. Alternatively, 
    # we could also use Jaccard or Jaro Winkler which calculates the percentage of matched characters.
    tfidf = TfidfVectorizer(min_df=1)
    fit = tfidf.fit_transform(input)
    sim = (fit * fit.T).A
    return (sim[0][1])

# ------------------------------------------------------------------------------------------
# This function takes string input and remove stop words and singularize the words.
def str_swr_sgl(str_input):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english") + list(punctuation))
    naked_words = [word for word in word_tokenize(str_input) if word not in stop_words]
    postag_words = pos_tag(naked_words)
    raw_words = [singularize(word[0]) for word in postag_words]
    return (raw_words)

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
    # Since the string has a clear structure with upper case letter followed by : for all of
    # important headers. We can use a simple regex
    regex = r"[A-Z / A-Z]{4,}\:"
    job_post = []
    df_ls = []
    match_headers = []
    match_items = []
    matches = re.finditer(regex, jobpost_str)
    
    # Loop through each matches and scrape the string starting from match.end to next match.start
    # e.g. TERM: Permanent until ..... POSITION:. It will grab the content after first ':' and
    # to the beginning of next match that is one index before P. Note that in this loop, we also
    # store all scraped content including those that we don't want like ANNOUNCE CODE:.
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
    # Here we extract our interested contents based on headers in clean_fields
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
            # This part is to skip the disclaimer sentence at the very end of year job post
            idx_ = res.find("--")
            if idx_ != -1:
                res = res[0:idx_].strip()
            res = res.replace("\r\n", "")
            df_ls.append(res)
        else:
            df_ls.append("NULL")
    return df_ls
        
# This function is a caller function to invoke jobpost related eda functions
# It will take a while to run this function due to heavy text analysis
def jobpost_eda():
    zf = zipfile.ZipFile("data_job_posts.zip")
    df = pd.read_csv(zf.open("data job posts.csv"))

    jobpost_company(df)
    jobpost_month(df)
    str_swr_sgl(df["JobRequirment"][0])
    n = len(df["jobpost"])

    # ----
    result = pd.DataFrame(columns = clean_fields)
    for i in range(0,n):
        temp = df["jobpost"][i]
        res = jobpost_scraper(temp)
        res[4] = str_swr_sgl(res[4])
        result.loc[i] = res
    print("Total {0} job posts scraped.".format(n))
    print(result[:6])
    # ----

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

# ------------------------------------------------------------------------------------------
# This is a simple model for fitting a binary classification model using the sign of 
# the slope. Additional rules can be added to further enhance this model such as if average
# sales volume is below certain number because emerging also means relatively unknown. Also,
# due to time contraint I assumed that the dataset fits the linear regression model. 
def is_emerging(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    if slope > 0:
        return True
    else:
        return False

# ------------------------------------------------------------------------------------------
# Most of the answers here can be found readily using DataFrame.describe()
# This function mainly aims to illustrate how it can be implemented.
def find_stats(input):
    max_val = max(input)
    min_val = min(input)
    n = len(input)
    mean_val = sum(input)/float(n)
    sorted_input = sorted(input)
    if n%2 == 0:
        idx1 = n/2
        idx2 = idx1 + 1
        median = (sorted_input[idx1] + sorted_input[idx2])/float(2)
    else:
        idx = (n+1)/float(2)
        median = sorted_input[idx]
    return [min_val, max_val, median, mean_val]

def is_outlier(value, qtr25, qtr75):
    iqr = qtr75 - qtr25
    lower = qtr25 - 1.5*iqr
    upper = qtr75 + 1.5*iqr
    return (value < lower) or (value > upper)

# This function is not used extensively but can be used to ignore non numeric values for summing
def audit_sum(input):
    total = []
    [total.append(int(i)) for i in input if i.isdigit()]
    return sum(total)

# ------------------------------------------------------------------------------------------
# All the questions answered in this function can be answered easily using pandas dataframe.
# Certain question is answered using DataFrame
# returning results as
# Dict - {
#  'Best Performing' : (product, total sales),
#  'Outliers' : {'P1' : [...], ...},
#  'Statistics' : {'P1' : [...], ...},
#  'Biweekly Worst' : {'P1' : [...], ...},
#  'Emerging Product' : [...]
# }
def sale_analysis():
    filename = "Sales_Transactions_Dataset_Weekly.csv"
    weeks = np.arange(1,53,1).tolist()
    biweekly = np.arange(1,53,2).tolist()
    df_bweek = pd.DataFrame(columns=["BW"+str(b) for b in biweekly])
    results = {}
    total_sales_by_prod = {}
    outliers = {}
    prod_stats = {}
    total_biweekly_sales = {}
    emerging_prod = []
    
    with open(filename, "r") as input_f:
        reader = csv.reader(input_f, delimiter=",")
        # Read first line as header
        header = next(reader)
        for r in reader:
            r_int = [int(r[i]) for i in range(1,53)]
            nr_fl = [float(r[i]) for i in range(55, len(r))]
            # Measure emerging trend of product based on positiveness of slope of linear regression
            if is_emerging(weeks, r_int):
                emerging_prod.append(r[0])
            # Calculates min,max,median,mean for each product
            prod_stats[r[0]] = find_stats(r_int)
            # Find outlier weekly sales for each product
            qtr25, qtr75 = np.percentile(r_int, [25,75])
            outlier_idx = [header[i] for i in range(1,53) if is_outlier(r_int[i-1], qtr25, qtr75)]
            outliers[r[0]] = outlier_idx
            # Sum biweekly sales over 52w period for each product
            df_bweek.loc[r[0],:] = [audit_sum(r[b:b+2]) for b in biweekly]
            # Sum sales volume over 52w period for each product
            total_sales_by_prod[r[0]] = sum(r_int)
    # Find biweekly worst performing products
    # The minimum for each biweekly sales is zero. On avg, there is ~150 products with zero biweekly sales.
    for d in df_bweek:
        bw_min = df_bweek.loc[:,d].min(axis=0)
        bw_min_p = df_bweek.loc[:,d] == bw_min
        total_biweekly_sales[d] = df_bweek[bw_min_p].index.tolist()
    # Sort for the product with best sales volume
    best_prod = max(total_sales_by_prod, key=total_sales_by_prod.get)
    # Store results into result dictionary
    results["Best Performing"] = (best_prod, total_sales_by_prod[best_prod])
    results["Outliers"] = outliers  
    results["Statistics"] = prod_stats
    results["Biweekly Worst"] = total_biweekly_sales  
    results["Emerging Product"] = emerging_prod
    return (results)

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
    # You need to input your kaggle accounts credentials for authentication
    login_url = "https://www.kaggle.com/account/login?isModal=true"
    payload =  {"username" : "_____", "password" : "_______", "__RequestVerificationToken" : ""}
    r = requests.session()
    result = r.get(login_url)
    payload["__RequestVerificationToken"] = r.cookies.get_dict()["__RequestVerificationToken"]
    result = r.post(login_url, data=payload, headers = dict(referer=login_url) )
    result = r.get(link,headers = dict(referer = link))

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
    print("------------------------- Q1 - Reverse sum")
    print("Index exclusion sum of {0}: {1}".format(input, val))

    # Question 2
    print("\n------------------------- Q2 - Running. This may take a while ...")
    results = sale_analysis()
    print("Best performing: ")
    print(results["Best Performing"])
    print("Outliers for product P8: ")
    print(results["Outliers"]["P8"])
    print("Statistics for product P8: ")
    print(results["Statistics"]["P8"])
    print("Biweekly worst for BW1 (W0+W1): ")
    print(results["Biweekly Worst"]["BW1"])
    print("Emerging products: ")
    print(results["Emerging Product"])

    # Question 3
    print("\n-------------------------Q3 - Running. This may take a while ..\n")
    jobpost_eda()

    # Question 4
    q4file = "test.csv"
    df = pd.read_csv(q4file)
    df["same_security"] = df.loc[:,["description_x", "description_y"]].apply(text_sim, axis=1)
    print("------------------------- Q4 - First 6 rows")
    print(" Similarity scores range between 0 to 1 with 1 means exactly identical\n")
    print(df.iloc[:6,1:])