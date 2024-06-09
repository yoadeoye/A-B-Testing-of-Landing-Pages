#!/usr/bin/env python
# coding: utf-8

# **A/B TESTING**: 
#     
#     could be named split testing, is a strategy used by data science professionals to determine 
#     which strategy is best to yield better outcome in terms of profit, better user engagement,sales,traffic,
#     more recongnition etc. 
# - Features 
# 
# 
# 1. Campaign Name: The name of the campaign
# 2. Date: Date of the record
# 3. Spend: Amount spent on the campaign in dollars
# 4. No of Impressions: Number of impressions the ad crossed through the campaign
# 4. Reach: The number of unique impressions received in the ad
# 5. No of Website Clicks: Number of website clicks received through the ads
# 6. No of Searches: Number of users who performed searches on the website 
# 7. No of View Content: Number of users who viewed content and products on the website
# 8. No of Add to Cart: Number of users who added products to the cart
# 9. No of Purchase: Number of purchases
# 
#    - Dataset:
# Two Ads campaigns were performed by the company:
# 
# 1. Control Campaign
# 2. Test Campaign
#    - Task : Perform A/B testing to find the best campaign for the company to get more customers.

# In[74]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from datetime import date,timedelta


# In[75]:


test_group=pd.read_csv("test_group.csv",sep=";")
control_group=pd.read_csv("control_group.csv",sep=";")


# In[76]:


print(test_group.head())
print(control_group.head())


# DATA PREPARATION AND FAMILIARIZATION

# In[77]:


#General Information on the two dATASET
control_group.info()
test_group.info()

# the two dataset are similar print("\n")


# In[78]:


#checking for null values
test_group.isna().sum() # NO NULL VALUE 


# In[79]:


#checking for null values
control_group.isna().sum() # all features except the name, date, spend, have null values == 7 columns with one null value each.


# In[80]:


# Renaming each column for more clarity
control_group.columns=["Campaign Name","Date","Amount Spent","nImpressions",
                      "nReach","Website Clicks","nSearch","nViewContent","nAddOfCart","nPurchase"]

test_group.columns=["Campaign Name","Date","Amount Spent","nImpressions",
                      "nReach","Website Clicks","nSearch","nViewContent","nAddOfCart","nPurchase"]


# In[81]:


# Handling missing values in control_group using Mean Imputation as its one in each feature 
control_group.nImpressions.fillna(control_group.nImpressions.mean(),inplace=True)

control_group.nReach.fillna(control_group.nReach.mean(),inplace=True)

control_group["Website Clicks"].fillna(control_group["Website Clicks"].mean(),inplace=True)

control_group.nSearch.fillna(control_group.nSearch.mean(),inplace=True)
control_group.nViewContent.fillna(control_group.nViewContent.mean(),inplace=True)
control_group.nAddOfCart.fillna(control_group.nAddOfCart.mean(),inplace=True)
control_group.nPurchase.fillna(control_group.nPurchase.mean(),inplace=True)


# In[82]:


#  The datasets are joined/merged to make up one A/b dataset
ab_data=control_group.merge(test_group,how="outer").sort_values(["Date"]).reset_index(drop=True)
ab_data.head(11) #prepared data for A/B Testing


# A/B TESTING TO DETERMINE THE BEST MARKETING STRATEGY i.e the best campaign name
# 
# - both campaign have the same no of samples (30)

# In[83]:


ab_data.describe() # statistical information of the numerical variables


# - Outliers are negligible as confirmed from Boxplot so we use pearson correlation

# In[84]:


ab_data.corr(method="pearson")["nPurchase"]


# - Only the number of users who added products to cart amongst other features show a slightly high positive correlation with the number of purchases made which is the target variable. This means that the observed probability that a change in number of users that added product to Cart could affect the number of purchases made is 38%. However further experimentation: Hypothesis testing needs to be performed further to confirm that.

# In[85]:


# Lets compare the no of impressions reached by each campaign in relation to the Amount Spent
pio.templates.default = "plotly_white"
import statsmodels
figure=px.scatter(data_frame=ab_data,x="nImpressions",y="Amount Spent",
                    size="Amount Spent",color="Campaign Name",trendline="ols")

figure.show()


# - The control campaign in relation to the AmOunt Spent shows the highest number of impressions

# In[86]:


# Comparing the no of Searches made by users in each campaign 
label=["Total Searches from Control Campaign","Total Searches from Test Campaign"]
counts=[sum(control_group.nSearch),sum(test_group.nSearch)]

# determine the larger portion to explode 
if counts[0]>counts[1]:
    explode=(0.05,0)
else:
    explode=(0,0.05)


colors=['light blue','orange']
figure=go.Figure(data=[go.Pie(labels=label, values=counts,pull=explode)])

figure.update_layout(title_text="Control Campaign Vs Test Campaign - No of Searches")
figure.update_traces(hoverinfo="label+percent",
                     textinfo='value',
                     textfont_size=30,
                     marker=dict(colors=colors,line=dict(color='black',width=2)))

figure.show()


# - From the chart above the no of searches made by users on the website in TEST Campaign is more than that of the Control Campaign by 4%
# 

# In[87]:


# Comparing the Website Clicks made by users in each campaign 
label=["Total Website Clicks from Control Campaign","Total Website Clicks from Test Campaign"]
counts=[sum(control_group['Website Clicks']),sum(test_group['Website Clicks'])]

# determine the larger portion to explode 
if counts[0]>counts[1]:
    explode=(0.05,0)
else:
    explode=(0,0.05)

colors=['light blue','orange']
figure=go.Figure(data=[go.Pie(labels=label, values=counts,pull=explode)])

figure.update_layout(title_text="Control Campaign Vs Test Campaign - Website Clicks")
figure.update_traces(hoverinfo="label+percent",
                     textinfo='value',
                     textfont_size=30,
                     marker=dict(colors=colors,line=dict(color='black',width=2)))

figure.show()


# - More Website Clicks made in TEST than in Control Campaign by 6%

# In[88]:


# Comparing the No of Content Viewed made by users in each campaign 
label=["No of Content Viewed from Control Campaign","No of Content Viewed from Test Campaign"]
counts=[sum(control_group.nViewContent),sum(test_group.nViewContent)]

# determine the larger portion to explode 
if counts[0]>counts[1]:
    explode=(0.05,0)
else:
    explode=(0,0.05)


colors=['light blue','orange']
figure=go.Figure(data=[go.Pie(labels=label, values=counts,pull=explode)])

figure.update_layout(title_text="Control Campaign Vs Test Campaign - no of Viewed Content")
figure.update_traces(hoverinfo="label+percent",
                     textinfo='value',
                     textfont_size=30,
                     marker=dict(colors=colors,line=dict(color='black',width=2)))

figure.show()


# - More contents were viewed on the CONTROL Campaign website as compared to the test campaign by only  2.2% .
# However, the test campaign generated more website clicks and more searches.
# 

# In[89]:


# Comparing the No of Products Added to Cart by users in each campaign 
label=["No of Products Added to Cart from Control Campaign","No of Products Added to Cart from Test Campaign"]
counts=[sum(control_group.nAddOfCart),sum(test_group.nAddOfCart)]

# determine the larger portion to explode 
if counts[0]>counts[1]:
    explode=(0.05,0)   # explode first one if its the largest
else:
    explode=(0,0.05)   # otherwise


colors=['light blue','orange']
figure=go.Figure(data=[go.Pie(labels=label, values=counts,pull=explode)])

figure.update_layout(title_text="Control Campaign Vs Test Campaign - No of Products Added to Cart")
figure.update_traces(hoverinfo="label+percent",
                     textinfo='value',
                     textfont_size=30,
                     marker=dict(colors=colors,line=dict(color='black',width=2)))

figure.show()


# - More products were added to cart from CONTROL Campaign compared to Test Campaign by a total of 12,554 units.

# In[90]:


# Comparing the Total Amount Spent foreach campaign 
label=["Total Amount Spent for Control Campaign","Total Amount Spent for Test Campaign"]
counts=[sum(control_group['Amount Spent']),sum(test_group['Amount Spent'])]

# determine the larger portion to explode 
if counts[0]>counts[1]:
    explode=(0.05,0) # explode first if its the largest
else:
    explode=(0,0.05) # otherwise


colors=['light blue','orange']
figure=go.Figure(data=[go.Pie(labels=label, values=counts,pull= explode)])

figure.update_layout(title_text="Control Campaign Vs Test Campaign - Total Amount Spent")
figure.update_traces(hoverinfo="label+percent",
                     textinfo='value',
                     textfont_size=30,
                     marker=dict(colors=colors,line=dict(color='black',width=2)))

figure.show()


# - More funds were spent in executing the TEST Campaign than the Control campaign by USD 8,293

# In[91]:


# Comparing the total purchases generated by each campaign 
label=["Total Purchases generated from Control Campaign","Total Purchases generated from Test Campaign"]
counts=[sum(control_group.nPurchase),sum(test_group.nPurchase)]

# determine the larger portion to explode 
if counts[0]>counts[1]:
    explode=(0.05,0) # explode first if its the largest
else:
    explode=(0,0.05)  # otherwise


colors=['light blue','orange']
figure=go.Figure(data=[go.Pie(labels=label, values=counts,pull=explode)])

figure.update_layout(title_text="Control Campaign Vs Test Campaign - Total Purchases generated")
figure.update_traces(hoverinfo="label+percent",
                     textinfo='value',
                     textfont_size=30,
                     marker=dict(colors=colors,line=dict(color='black',width=2)))

figure.show()


# - The resulting purchases which is the target of the company reveals that the CONTROL campaign generated more purchases than the TEST campaign by only a negligible amount of 46 units which is less than 1%.
# - Thus we can conclude that the CONTROL campaign outperform the TEST ad campaign because less amount of money were spent,more purchases were made, more contents were viewed and more products were added to cart. These are the major metrics that will improve the company's sales.

# In[92]:


ab_data.corr()


# From the correlation between each Numerical features/metrics as observed, It could be detected that some significant relationships do does exist between: 
# 
# - nImpressions vs nReach	
# - nPurchase  vs nAddOfCart
# - nSearch vs 	nViewContent
# - Website Clicks vs nViewContent
#     - Now let's explore the metrics to decide which ad campaign converts more:

# In[93]:


#  Number of Impressions Vs No of Reach

figure=px.scatter(x='nReach',
                  y='nImpressions',
                  data_frame=ab_data, 
                  color='Campaign Name',
                  size='nImpressions',
                  trendline='ols')
figure.update_layout(title_text=" No of Reach Vs No of Impressions - Per Campaign")

figure.show()


# - The CONTROL CAMPAIGN has more conversion rate than the TEST CAMPAIGN in terms of the no of impressions and reach.
# - For faster result, Test Campaign will be any company's choice as it quickly starts yielding result in terms of impression and reach 

# In[94]:


# Website Clicks Vs No of Viewed Content.

figure=px.scatter(x='nViewContent',
                  y='Website Clicks',
                  data_frame=ab_data, 
                  color='Campaign Name',
                  size='Website Clicks',
                  trendline='ols')
figure.update_layout(title_text="  Website Clicks Vs No of Viewed Content  - Per Campaign")

figure.show()


# - the TEST Campaign has the highest number of website clicks but the CONTROL has the highest number of content viewed thus the CONTROL converts more.

# In[95]:


# No of Viewed Content Vs no of Search- per Campaign
figure=px.scatter(x='nSearch',
                  y='nViewContent',
                  data_frame=ab_data,
                  size='nSearch',
                  color='Campaign Name',
                  trendline='ols')

figure.update_layout(title_text='No of Viewed Content Vs no of Search- per Campaign')
figure.show()


# - CONTROL CAMPAIGN WINS HERE AS WELL!

# A product only has the chance of being purchased if it has been added to Cart. Let us explore the rate at which it occurs for each Campaign

# In[96]:


# nPurchase  vs nAddOfCart
figure=px.scatter(x='nPurchase',
                 y='nAddOfCart',
                 data_frame=ab_data,
                 size='nPurchase',
                 color='Campaign Name',
                 trendline='ols')

figure.update_layout(title_text=' Added To Cart  Vs No of Purchase- per Campaign')
figure.show()

Upon analysis of the data, it could be observed that the CONTROL campaign exhibits a higher volume of products being added to the cart compared to the TEST campaign.

However, the rate at which the product added are being purchased is higher in TEST than CONTROL CAMPAIGN.
Control campaign performs better than the test as it brings about more sales in total, more searches, engagement and more products Added to Cart. But the Conversion rate for TEST is higher. 

Given these insights, a business could channel TEST campaign towards a specific audience and products for  quick positive outcome whilst control campaign could be channelled towards numerous audience and many products. Overall, both approaches a business can optimize campaign deployment to determine the target market and audience. 

# #### FURTHER ANALYSIS FOR MORE INSIGHTS 
# ###### CLICK THROUGH RATE  OF EACH CAMPAIGN ADS

# + The Click Through Rate For Each Campaign which is calculated as a percentage of the ratio of Website Click/No of Reach.
# 
# 
# + CTR = (∑ WEBSITE CLICKS/ ∑REACH) * 100
# 
# 
# + we will compute the Empirical Cummulative Distibution of each campaign CTR to observe any difference in the CTR and its magnitude.
# 

# In[97]:


sns.set_style("whitegrid")   # sets the white background plot.

# sets the figure
fig, ax = plt.subplots(1,3, figsize=(18,10), sharex=False)

#plots the empirical cummulative distribution function plot for REACH.

sns.ecdfplot(x="nReach",
             data=ab_data,
             hue="Campaign Name",
             palette="Dark2", ax=ax[0],marker=True)

ax[0].set_title("ECDF for Campaign Name - No of Reach",fontsize=15,fontweight='bold')
ax[0].set_xlabel("No of Reach", fontsize=12,fontweight='bold')
ax[0].set_ylabel("ECDF",fontsize=12,fontweight='bold')


# ECDF for WEBSITE CLICKS
sns.ecdfplot(x="Website Clicks",data=ab_data,hue="Campaign Name",palette='Dark2',ax=ax[1],marker=True)

ax[1].set_title("ECDF for Campaign Name - Website Clicks",fontsize=15,fontweight='bold')
ax[1].set_xlabel("Website Clicks", fontsize=12,fontweight='bold')
ax[1].set_ylabel("ECDF ",fontsize=12,fontweight='bold')



#lets compute the Click Through Rate Using Formula above
CTR=ab_data.groupby(['Campaign Name'])['Website Clicks'].sum()/ab_data.groupby(['Campaign Name'])['nReach'].sum()*100

# lets plot the CTR
colors=['blue','orange']
#adjust color,set the axis,set the position of xlabels 
CTR.plot(kind='bar',ax=ax[2],rot = 0,width = 0.80,alpha = 0.9,fontsize = 12,color=colors)

# customizing the plot to show percentage of CTR per Campaign
for i, g in enumerate(CTR):
    ax[2].text(i, g - 3, "{0:.{digits}f}%".format(g, digits=2), color='white',
               fontsize=19, fontweight="bold", ha="center", va='center')

# customize and label the axis
ax[2].set_title("CTR for Campaign Name",fontsize=15,fontweight='bold')
ax[2].set_xlabel("Campaign Name", fontsize=12,fontweight='bold')
ax[2].set_ylabel("CTR",fontsize=12,fontweight='bold')

plt.show()


# ###### INSIGHT
# - The ECDF measures the overall distribution of the data and it displays percentile easily on the y-axis. The plot
# - above shows the count/proportion by which a data point falls below a unique value in a dataset x P(x<=x).
# - Here the no of reaches generated by the Control Campaign is a little higher than the Test Campaign. The cdf plot shows that the test campaign cdf is 6 times(@ 6 percentiles ) equal to the cdf of the control campaign and less 7 times(@7 percentile).
# - For the website clicks,the probabibility that it is greater in Test Campaign is higher than Control Campaign. 
# - The dataset is filled with discrete values thus the reason for steeps.
# - Finally the Click Through Rate  is higher for Test than Control Campaign - this confirms that Test campaign has higher conversion rate than Control Campaign ads.
# 

# ###### TIME SERIES ANALYSIS IN RELATION TO AMOUNT SPEND AND CLICK THROUGH RATE

# In[98]:


ab_data.head()


# In[99]:



ab_data['Date']=pd.to_datetime(ab_data['Date'],format='%d.%m.%Y')  # convert the 'Date' column to datetime format
ab_data['Day'] = ab_data['Date'].dt.day  # extract the month from the 'Date' column
ab_data = ab_data.sort_values(by='Day')  # sort the DataFrame by 'Month' in ascending order
ab_data.head()


# In[101]:


sns.set_style('whitegrid')
# compute and add the click through rate to the dataset
ab_data['CTR']= ab_data['Website Clicks']/ab_data.nReach

# group the Click Through Rate for both campaigns with the date 
all_ctr=ab_data.groupby(['Campaign Name','Date'])['CTR'].sum()

# group the Amount Spend(USD)for both campaigns with the Date
all_spend=ab_data.groupby(['Campaign Name','Date'])['Amount Spent'].sum()

# plot the CLick Through Rate with the Date for each campaign ads
fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(18,20))

#  plot the CLick Through Rate per Date for the control campaign 
all_ctr['Control Campaign'].plot(marker='*',ls='-',markersize=7, alpha=1,fontsize=12, label='Control Campaign',ax=ax[0])
all_ctr['Test Campaign'].plot(marker='o',ls='--',markersize=7, alpha=1,fontsize=12, label='Test Campaign',ax=ax[0])

ax[0].set_title("The Click Through Rate Trend Per Day for Each Campaign ",fontsize=12,fontweight='bold')
ax[0].set_xlabel("Date of The Month",fontsize=12,fontweight='bold')
ax[0].set_ylabel("Click Through Rate ",fontsize=12,fontweight='bold')

ax[0].legend(fontsize=14)


#plot on the same graph the Amount Spend per day for  the test campaign
all_spend['Control Campaign'].plot(marker='*',ls='-',markersize=7,alpha=1,fontsize=12,label='Control Campaign',ax=ax[1])
all_spend['Test Campaign'].plot(marker='o',ls='--',markersize=7,alpha=1,fontsize=12,label='Test Campaign',ax=ax[1])

ax[1].set_title("The Amount Spent Trend Per Day for Control Campaign ",fontsize=12,fontweight='bold')
ax[1].set_xlabel("Date of The Month",fontsize=12,fontweight='bold')
ax[1].set_ylabel("Amount Spent-USD",fontsize=12,fontweight='bold')


ax[1].legend(fontsize=14)


# - From above CTR trend, the Campaign ads happened in the month of August and The CTR for Test campaign is Bimodal on 12th and 19th of August and then falls afterwards while that of the Control Campaign show a steady trend throughout the month
# 
# - From the Amount Trend, more funds was spent for the Test Campaign within the 3rd week, thus we can conclude the bimodal CTR trend was as a result of more money pushed for Test Campaign.
# 

# #### HYPOTHESIS TESTING 
# - Ho/Null Hypothesis: No statistically significant difference between the 2 campaigns
# - Ha/Alternative hypothesis: Statistically significant difference does exist between the 2 campaigns.
# - With alpha =0.05, if p-value < alpha, Hi -is true and Ho is false and vice-versa
# 

# In[102]:


# Compute the Conversion Rate: the ratio of the No of products added to Cart/No of Purchases Made
ab_data['conv_rate']=ab_data.nAddOfCart/ab_data.nPurchase


# In[104]:


from scipy.stats import ttest_ind
def campaign_test(df,col):
    #getting the features value in the data
    control=df[df["Campaign Name"]=="Control Campaign"][col]
    test=df[df["Campaign Name"]=="Test Campaign"][col]
    # hypothesis testing- 2 sample t-test
    t_stat_,p_val_=ttest_ind(control,test,equal_var=False)
    return t_stat_,p_val_

# Nummerical features selected for hypothesis testing
num_feat=ab_data.select_dtypes(exclude='object').drop(columns=['Date','Day','nReach','Website Clicks'],
                                                      axis=1).columns.to_list()
for i in num_feat:
    print(f"For {i}, the t_stat and p_value are :  {campaign_test(ab_data,i)}\n")


# In[105]:


comparison_table=pd.DataFrame({"Metrics": num_feat,
                               "T-statistics": [campaign_test(ab_data,i)[0] for i in num_feat],
                               "P-value": [campaign_test(ab_data,i)[1] for i in num_feat],
                               "P-value < α ": [True if campaign_test(ab_data,i)[1] < 0.05 else False for i in num_feat],
                               "Hypotheis": ['Reject Ho/Accept Ha' if campaign_test(ab_data,i)[1] < 0.05 else 'Reject Ha/Accept Ho' for i in num_feat],
                               "Inference": ['significant diff' if campaign_test(ab_data,i)[1] < 0.05 else 'No significant diff' for i in num_feat]
                               
                              })
print("The result of The Statistical Testing(two sample T-Test) based on significance level, alpha=0.05")
comparison_table


# In[ ]:





# In[ ]:




