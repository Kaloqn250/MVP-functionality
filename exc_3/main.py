import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('machine_learning/exc_3/train.csv')
df2 = pd.read_csv('machine_learning/exc_3/test.csv')



df = df.drop(['bdate', 'city', 'has_photo', 'has_mobile', 'education_status', 'followers_count', 'relation', 'graduation', 'career_start', 'career_end', 'last_seen', 'occupation_name', 'langs'], axis=1)
df2 = df2.drop(['bdate', 'city', 'has_photo', 'has_mobile', 'education_status', 'followers_count', 'relation', 'graduation', 'career_start', 'career_end', 'last_seen', 'occupation_name', 'langs'], axis=1)


def fill_education_form(row):
    if pd.isnull(row['education_form']):
        return 'Distance Learning'
    
    return row['education_form']


def fill_occupation_type(row):
    if pd.isnull(row['occupation_type']):
        return 'university'
    
    return row['occupation_type']


def change_education_form(row):
    # education forms
    # Full-time - 0
    # Distance Learning - 1
    # Part-time - 2

    if row['education_form'] == 'Full-time':
        return 0
    
    elif row['education_form'] == 'Distance Learning':
        return 1
    
    elif row['education_form'] == 'Part-time':
        return 2


def change_life_main(row):
    if row['life_main'] == 'False':
        return 0
    
    return int(row['life_main'])


def change_people_main(row):
    if row['people_main'] == 'False':
        return 0
    
    return int(row['people_main'])


def change_occupation_type(row):
    # university - 0
    # work - 1

    if row['occupation_type'] == 'university':
        return 0

    return 1

df['education_form'] = df.apply(fill_education_form, axis=1)
df['occupation_type'] = df.apply(fill_occupation_type, axis=1)
df['education_form'] = df.apply(change_education_form, axis=1)
df['life_main'] = df.apply(change_life_main, axis=1)
df['people_main'] = df.apply(change_people_main, axis=1)
df['occupation_type'] = df.apply(change_occupation_type, axis=1)


df2['education_form'] = df2.apply(fill_education_form, axis=1)
df2['occupation_type'] = df2.apply(fill_occupation_type, axis=1)
df2['education_form'] = df2.apply(change_education_form, axis=1)
df2['life_main'] = df2.apply(change_life_main, axis=1)
df2['people_main'] = df2.apply(change_people_main, axis=1)
df2['occupation_type'] = df2.apply(change_occupation_type, axis=1)



X = df.drop(['result'], axis=1)
Y = df['result']


model = LogisticRegression(max_iter=10000)

model.fit(X, Y)

y_pred = model.predict(df2)

ID = df2['id']


df_result = pd.DataFrame({
    'id': ID,
    'result': y_pred
}) 

df_result.to_csv('result.csv', index=False)