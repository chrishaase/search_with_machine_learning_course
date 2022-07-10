import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data_t10000_stem.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])
parentDict = parents_df.set_index('category').to_dict()['parent']
#print(parents_df.head())

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]
queryDict1 = df.set_index('query').to_dict()['category']
print(list(queryDict1.items())[:5])

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queryDict2 = {k.lower():v for k,v in queryDict1.items()}
queryDict3 = {}
for k,v in queryDict2.items():
    words = k.split()   
    stemmedwords = []
    for word in words:
        stemmedwords.append(stemmer.stem(word))
    queryDict3[" ".join(stemmedwords)] = v
    print(" ".join(stemmedwords))
queryDict = queryDict3
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
threshold = 10000
occur = df.groupby(['category']).size().to_frame('size').reset_index()
resultSize = occur['size'].sum()
print("size initial", resultSize)
dictOccur = occur.set_index('category').to_dict()['size']
dictResult = {}
done = False
print("threshold set to ", threshold)
while not done:
    copy = {k:v for k,v in dictOccur.items()}
    for k,v in dictOccur.items():
        print(k,v)
        if v > threshold:
            if k not in dictResult:
                dictResult[k] = v
            else:
                dictResult[k] += v
        elif k in parentDict:
            if parentDict[k] not in copy:
                copy[parentDict[k]] = v
            else:
                copy[parentDict[k]] += v
            # replace cat k with parentCat[k] in queryDict
            queryDict = {query:(parentDict[k] if cat == k else cat) for query, cat in queryDict.items()}
        copy.pop(k, None)
    #print(len(copy))
    if len(copy) == 0:
        done = True
    dictOccur = {k:v for k,v in copy.items()}
resultSize2 = np.sum(list(dictResult.values()))
print("final resultSize - without missing values", resultSize2)
print("final amount of categories", len(dictResult))
df = pd.DataFrame.from_dict(queryDict, orient="index", columns=['category']).rename_axis('query').reset_index()
df['query']=df['query'].str.lower()
#print(df.head())
# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
