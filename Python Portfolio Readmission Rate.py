#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# In[4]:


data_2015 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Readmissions and Deaths - Hospital_2015.csv', engine='python')
data_2016 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Readmissions and Deaths - Hospital_2016.csv', engine='python')
data_2017 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Hospital Returns - Hospital_2017.csv', engine='python')
data_2018 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Unplanned Hospital Visits - Hospital_2018.csv', engine='python')
data_2019 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Unplanned Hospital Visits - Hospital_2019.csv', engine='python')
data_2020 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Unplanned Hospital Visits - Hospital_2020.csv', engine='python')
data_2021 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Unplanned_Hospital_Visits-Hospital_2021.csv', engine='python')
data_2022 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Unplanned_Hospital_Visits-Hospital_2022.csv', engine='python')
data_2023 = pd.read_csv(r'C:\Users\Ankit\Documents\George Mason Grad\Spring 2024\HAP 725 Case Study\Complications & Deaths + Unplanned Hosp Visit\Unplanned_Hospital_Visits-Hospital_2023.csv', engine='python')
data_2018.head()


# In[5]:


data_2015_filtered = pd.DataFrame(data_2015, columns = ['Provider ID', 'Hospital Name', 'City', 'Measure ID', 'Score', 'Measure Start Date', 'Measure End Date'])
data_2015_filtered = data_2015_filtered[data_2015_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2016_filtered = pd.DataFrame(data_2016, columns = ['Provider ID', 'Hospital Name', 'City', 'Measure ID', 'Score', 'Measure Start Date', 'Measure End Date'])
data_2016_filtered = data_2016_filtered[data_2016_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2017_filtered = pd.DataFrame(data_2017, columns = ['Provider ID', 'Hospital Name', 'City', 'Measure ID', 'Score', 'Measure Start Date', 'Measure End Date'])
data_2017_filtered = data_2017_filtered[data_2017_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2018_filtered = pd.DataFrame(data_2018, columns = ['Provider ID', 'Hospital Name', 'City', 'Measure ID', 'Score', 'Measure Start Date', 'Measure End Date'])
data_2018_filtered = data_2018_filtered[data_2018_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2019_filtered = pd.DataFrame(data_2019, columns = ['Facility ID', 'Facility Name', 'City', 'Measure ID', 'Score', 'Start Date', 'End Date'])
data_2019_filtered = data_2019_filtered[data_2019_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2020_filtered = pd.DataFrame(data_2020, columns = ['Facility ID', 'Facility Name', 'City', 'Measure ID', 'Score', 'Start Date', 'End Date'])
data_2020_filtered = data_2020_filtered[data_2020_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2021_filtered = pd.DataFrame(data_2021, columns = ['Facility ID', 'Facility Name', 'City', 'Measure ID', 'Score', 'Start Date', 'End Date'])
data_2021_filtered = data_2021_filtered[data_2021_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2022_filtered = pd.DataFrame(data_2022, columns = ['Facility ID', 'Facility Name', 'City', 'Measure ID', 'Score', 'Start Date', 'End Date'])
data_2022_filtered = data_2022_filtered[data_2022_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2023_filtered = pd.DataFrame(data_2023, columns = ['Facility ID', 'Facility Name', 'City', 'Measure ID', 'Score', 'Start Date', 'End Date'])
data_2023_filtered = data_2023_filtered[data_2023_filtered['Measure ID'].isin(['READM_30_AMI'])]

data_2023_filtered.head()


# In[6]:


data_2015_filtered['Score'] = pd.to_numeric(data_2015_filtered['Score'], errors='coerce')
data_2015_filtered = data_2015_filtered.replace(np.nan, 0, regex=True)

data_2016_filtered['Score'] = pd.to_numeric(data_2016_filtered['Score'], errors='coerce')
data_2016_filtered = data_2016_filtered.replace(np.nan, 0, regex=True)

data_2017_filtered['Score'] = pd.to_numeric(data_2017_filtered['Score'], errors='coerce')
data_2017_filtered = data_2017_filtered.replace(np.nan, 0, regex=True)

data_2018_filtered['Score'] = pd.to_numeric(data_2018_filtered['Score'], errors='coerce')
data_2018_filtered = data_2018_filtered.replace(np.nan, 0, regex=True)

data_2019_filtered['Score'] = pd.to_numeric(data_2019_filtered['Score'], errors='coerce')
data_2019_filtered = data_2019_filtered.replace(np.nan, 0, regex=True)

data_2020_filtered['Score'] = pd.to_numeric(data_2020_filtered['Score'], errors='coerce')
data_2020_filtered = data_2020_filtered.replace(np.nan, 0, regex=True)

data_2021_filtered['Score'] = pd.to_numeric(data_2021_filtered['Score'], errors='coerce')
data_2021_filtered = data_2021_filtered.replace(np.nan, 0, regex=True)

data_2022_filtered['Score'] = pd.to_numeric(data_2022_filtered['Score'], errors='coerce')
data_2022_filtered = data_2022_filtered.replace(np.nan, 0, regex=True)

data_2023_filtered['Score'] = pd.to_numeric(data_2023_filtered['Score'], errors='coerce')
data_2023_filtered = data_2023_filtered.replace(np.nan, 0, regex=True)


# In[7]:


Hosp = ['SENTARA NORFOLK GENERAL HOSPITAL', 'BON SECOURS DEPAUL MEDICAL CENTER', 'SENTARA LEIGH HOSPITAL', 'LEWISGALE MEDICAL CENTER']

city = ['NORFOLK', 'SALEM']

data_2015_filtered2 = pd.DataFrame(data_2015_filtered[(data_2015_filtered['Hospital Name'].isin(Hosp))])
data_2015_filtered2 = data_2015_filtered2[(data_2015_filtered2['City'].isin(city))]
print("\033[1m" + '2015 Filtered Dataset' + "\033[0m")
print(data_2015_filtered2)

data_2016_filtered2 = pd.DataFrame(data_2016_filtered[(data_2016_filtered['Hospital Name'].isin(Hosp))])
data_2016_filtered2 = data_2016_filtered2[(data_2016_filtered2['City'].isin(city))]
print("\033[1m" + '2016 Filtered Dataset' + "\033[0m")
print(data_2016_filtered2)

data_2017_filtered2 = pd.DataFrame(data_2017_filtered[(data_2017_filtered['Hospital Name'].isin(Hosp))])
data_2017_filtered2 = data_2017_filtered2[(data_2017_filtered2['City'].isin(city))]
print("\033[1m" + '2017 Filtered Dataset' + "\033[0m")
print(data_2017_filtered2)

data_2018_filtered2 = pd.DataFrame(data_2018_filtered[(data_2018_filtered['Hospital Name'].isin(Hosp))])
data_2018_filtered2 = data_2018_filtered2[(data_2018_filtered2['City'].isin(city))]
print("\033[1m" + '2018 Filtered Dataset' + "\033[0m")
print(data_2018_filtered2)

data_2019_filtered2 = pd.DataFrame(data_2019_filtered[(data_2019_filtered['Facility Name'].isin(Hosp))])
data_2019_filtered2 = data_2019_filtered2[(data_2019_filtered2['City'].isin(city))]
print("\033[1m" + '2019 Filtered Dataset' + "\033[0m")
print(data_2019_filtered2)

data_2020_filtered2 = pd.DataFrame(data_2020_filtered[(data_2020_filtered['Facility Name'].isin(Hosp))])
data_2020_filtered2 = data_2020_filtered2[(data_2020_filtered2['City'].isin(city))]
print("\033[1m" + '2020 Filtered Dataset' + "\033[0m")
print(data_2020_filtered2)

data_2021_filtered2 = pd.DataFrame(data_2021_filtered[(data_2021_filtered['Facility Name'].isin(Hosp))])
data_2021_filtered2 = data_2021_filtered2[(data_2021_filtered2['City'].isin(city))]
print("\033[1m" + '2021 Filtered Dataset' + "\033[0m")
print(data_2021_filtered2)

data_2022_filtered2 = pd.DataFrame(data_2022_filtered[(data_2022_filtered['Facility Name'].isin(Hosp))])
data_2022_filtered2 = data_2022_filtered2[(data_2022_filtered2['City'].isin(city))]
print("\033[1m" + '2022 Filtered Dataset' + "\033[0m")
print(data_2022_filtered2)

data_2023_filtered2 = pd.DataFrame(data_2023_filtered[(data_2023_filtered['Facility Name'].isin(Hosp))])
data_2023_filtered2 = data_2023_filtered2[(data_2023_filtered2['City'].isin(city))]
print("\033[1m" + '2023 Filtered Dataset' + "\033[0m")
print(data_2023_filtered2)


# In[8]:


print(data_2015_filtered2.shape)
print(data_2016_filtered2.shape)
print(data_2017_filtered2.shape)
print(data_2018_filtered2.shape)
print(data_2019_filtered2.shape)
print(data_2020_filtered2.shape)
print(data_2021_filtered2.shape)
print(data_2022_filtered2.shape)


# In[9]:


hosp = ['SENTARA NORFOLK GENERAL HOSPITAL']

data_2015_Nor = pd.DataFrame(data_2015_filtered2[(data_2015_filtered2['Hospital Name'].isin(hosp))])
avgscore_Nor_2015 = data_2015_Nor['Score'].mean()

data_2016_Nor = pd.DataFrame(data_2016_filtered2[(data_2016_filtered2['Hospital Name'].isin(hosp))])
avgscore_Nor_2016 = data_2016_Nor['Score'].mean()

data_2017_Nor = pd.DataFrame(data_2017_filtered2[(data_2017_filtered2['Hospital Name'].isin(hosp))])
avgscore_Nor_2017 = data_2017_Nor['Score'].mean()

data_2018_Nor = pd.DataFrame(data_2018_filtered2[(data_2018_filtered2['Hospital Name'].isin(hosp))])
avgscore_Nor_2018 = data_2018_Nor['Score'].mean()

data_2019_Nor = pd.DataFrame(data_2019_filtered2[(data_2019_filtered2['Facility Name'].isin(hosp))])
avgscore_Nor_2019 = data_2019_Nor['Score'].mean()

data_2020_Nor = pd.DataFrame(data_2020_filtered2[(data_2020_filtered2['Facility Name'].isin(hosp))])
avgscore_Nor_2020 = data_2020_Nor['Score'].mean()

data_2021_Nor = pd.DataFrame(data_2021_filtered2[(data_2021_filtered2['Facility Name'].isin(hosp))])
avgscore_Nor_2021 = data_2021_Nor['Score'].mean()

data_2022_Nor = pd.DataFrame(data_2022_filtered2[(data_2022_filtered2['Facility Name'].isin(hosp))])
avgscore_Nor_2022 = data_2022_Nor['Score'].mean()

print(avgscore_Nor_2015)
print(avgscore_Nor_2016)
print(avgscore_Nor_2017)
print(avgscore_Nor_2018)
print(avgscore_Nor_2019)
print(avgscore_Nor_2020)
print(avgscore_Nor_2021)
print(avgscore_Nor_2022)


# In[10]:


data_2015_Comp = pd.DataFrame(data_2015_filtered2[(data_2015_filtered2['Hospital Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2015 = data_2015_Comp['Score'].mean()

data_2016_Comp = pd.DataFrame(data_2016_filtered2[(data_2016_filtered2['Hospital Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2016 = data_2016_Comp['Score'].mean()

data_2017_Comp = pd.DataFrame(data_2017_filtered2[(data_2017_filtered2['Hospital Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2017 = data_2017_Comp['Score'].mean()

data_2018_Comp = pd.DataFrame(data_2018_filtered2[(data_2018_filtered2['Hospital Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2018 = data_2018_Comp['Score'].mean()

data_2019_Comp = pd.DataFrame(data_2019_filtered2[(data_2019_filtered2['Facility Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2019 = data_2019_Comp['Score'].mean()

data_2020_Comp = pd.DataFrame(data_2020_filtered2[(data_2020_filtered2['Facility Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2020 = data_2020_Comp['Score'].mean()

data_2021_Comp = pd.DataFrame(data_2021_filtered2[(data_2021_filtered2['Facility Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2021 = data_2021_Comp['Score'].mean()

data_2022_Comp = pd.DataFrame(data_2022_filtered2[(data_2022_filtered2['Facility Name'] !='SENTARA NORFOLK GENERAL HOSPITAL')])
avgscore_Comp_2022 = data_2022_Comp['Score'].mean()

print(avgscore_Comp_2015)
print(avgscore_Comp_2016)
print(avgscore_Comp_2017)
print(avgscore_Comp_2018)
print(avgscore_Comp_2019)
print(avgscore_Comp_2020)
print(avgscore_Comp_2021)
print(avgscore_Comp_2022)


# In[11]:


df_all = [data_2015_filtered2['Score'], data_2016_filtered2['Score'], data_2017_filtered2['Score'], data_2018_filtered2['Score'], data_2019_filtered2['Score'], data_2020_filtered2['Score'], data_2021_filtered2['Score'], data_2022_filtered2['Score']]

headers = ['AllScore2015', 'AllScore2016', 'AllScore2017', 'AllScore2018', 'AllScore2019', 'AllScore2020', 'AllScore2021', 'AllScore2022']

df1_all = pd.concat(df_all, axis=1, keys=headers)

df1_all.head()


# In[12]:


#Grand Avg
avgall = df1_all.stack().mean()
avgall


# In[13]:


stdall = df1_all.stack().std()
stdall


# In[14]:


df = [data_2015_Comp['Score'], data_2016_Comp['Score'], data_2017_Comp['Score'], data_2018_Comp['Score'], data_2019_Comp['Score'], data_2020_Comp['Score'], data_2021_Comp['Score'], data_2022_Comp['Score']]

headers = ['Score2015', 'Score2016', 'Score2017', 'Score2018', 'Score2019', 'Score2020', 'Score2021', 'Score2022']

df1 = pd.concat(df_all, axis=1, keys=headers)

df1.head()


# In[15]:


avg2015 = df1['Score2015'].mean()
avg2016 = df1['Score2016'].mean()
avg2017 = df1['Score2017'].mean()
avg2018 = df1['Score2018'].mean()
avg2019 = df1['Score2019'].mean()
avg2020 = df1['Score2020'].mean()
avg2021 = df1['Score2021'].mean()
avg2022 = df1['Score2022'].mean()

print(avg2015)
print(avg2016)
print(avg2017)
print(avg2018)
print(avg2019)
print(avg2020)
print(avg2021)
print(avg2022)


# In[16]:


n_2015=int(df1['Score2015'].count())
n_2016=int(df1['Score2016'].count())
n_2017=int(df1['Score2017'].count())
n_2018=int(df1['Score2018'].count())
n_2019=int(df1['Score2019'].count())
n_2020=int(df1['Score2020'].count())
n_2021=int(df1['Score2021'].count())
n_2022=4

print(n_2015)
print(n_2016)
print(n_2017)
print(n_2018)
print(n_2019)
print(n_2020)
print(n_2021)
print(n_2022)


# In[17]:


zstat = 1.96


# In[18]:


std2015 = stdall/(math.sqrt(n_2015))
std2016 = stdall/(math.sqrt(n_2016))
std2017 = stdall/(math.sqrt(n_2017))
std2018 = stdall/(math.sqrt(n_2018))
std2019 = stdall/(math.sqrt(n_2019))
std2020 = stdall/(math.sqrt(n_2020))
std2021 = stdall/(math.sqrt(n_2021))
std2022 = stdall/(math.sqrt(n_2022))

print(std2015)
print(std2016)
print(std2017)
print(std2018)
print(std2019)
print(std2020)
print(std2021)
print(std2022)


# In[19]:


df3 = [['2015', avgscore_Nor_2015, avg2015, std2015, avgall, zstat],
      ['2016', avgscore_Nor_2016, avg2016, std2016, avgall, zstat],
      ['2017', avgscore_Nor_2017, avg2017, std2017, avgall, zstat],
      ['2018', avgscore_Nor_2018, avg2018, std2018, avgall, zstat],
      ['2019', avgscore_Nor_2019, avg2019, std2019, avgall, zstat],
      ['2020', avgscore_Nor_2020, avg2020, std2020, avgall, zstat],
      ['2021', avgscore_Nor_2021, avg2021, std2021, avgall, zstat],
      ['2022', avgscore_Nor_2022, avg2022, std2022, avgall, zstat]]

df3


# In[20]:


df4 = pd.DataFrame(df3, columns = ['Year', 'SENTARA NORFOLK GENERAL HOSPITAL', 'Avg Readmission Rate Comp Hosp', 'StdPerTime', 'Avg Readmission Rate All', 'zstat'])
df4


# In[21]:


df4['UCL for Competitor Hospitals'] = df4['Avg Readmission Rate All'] + df4['zstat']*df4['StdPerTime']
df4['LCL for Competitor Hospitals'] = df4['Avg Readmission Rate All'] - df4['zstat']*df4['StdPerTime']
df4


# In[23]:


plt.plot('Year', 'SENTARA NORFOLK GENERAL HOSPITAL', data=df4, marker='s', markerfacecolor='blue', linewidth=1, color='blue')
plt.plot('Year', 'UCL for Competitor Hospitals', data=df4, markersize=0, color='red', linewidth=1)
plt.plot('Year', 'LCL for Competitor Hospitals', data=df4, markersize=0, color='red', linewidth=1)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', borderaxespad=0)
plt.title('Sentara Norfolk VS VA Competitors - Heart Attack Readmission Rate')
plt.xlabel('Year')
plt.ylabel('Readmission Rate')


# In[ ]:




