#!/usr/bin/env python
# coding: utf-8

# #### An online bookstore is looking to optimize its website design to improve user engagement and ultimately increase book purchases.
#  The website currently offers two themes for its users: “Light Theme” and “Dark Theme.”
#   The bookstore’s data science team wants to conduct an A/B testing experiment to determine which theme leads to better user engagement and higher conversion rates for book purchases.
# 
# The data collected by the bookstore contains user interactions and engagement metrics for both the Light Theme and Dark Theme.
#  The dataset includes the following key features:
# 
# Theme: dark or light
# + Click Through Rate: The proportion of the users who click on links or buttons on the website.
# + Conversion Rate: The percentage of users who signed up on the platform after visiting for the first time.
# + Bounce Rate: The percentage of users who leave the website without further interaction after visiting a single page.
# + Scroll Depth: The depth to which users scroll through the website pages.
# + Age: The age of the user.
# + Location: The location of the user.
# + Session Duration: The duration of the user’s session on the website.
# + Purchases: Whether the user purchased the book (Yes/No).
# + Added_to_Cart: Whether the user added books to the cart (Yes/No).
# 
# Your task is to identify which theme, Light Theme or Dark Theme, yields better user engagement, purchases and conversion rates.
#  You need to determine if there is a statistically significant difference in the key metrics between the two themes.
# 
# + SITUATION: An online bookstore is faced with making a business decision concerning its web design optimization.
# The web design must be able to improve user engagement and drive sales.
# + TASK : To identify which theme Light or Dark will solve the problem
# + ACTION: To conduct an A/B Testing of to determine which theme based on the features highlighted above will encourage book purchase.
# + RESULT:
!pip install scipy
# In[100]:


import pandas as pd
from scipy.stats import ttest_ind
data=pd.read_csv("website_ab_test.csv")
data


# In[101]:


print("Statistical Description of the Numerical Features\n")
data.describe()


# 
# *Click Through Rate*: Ranges from 0.01 to 0.50 with a mean and std: of 0.26 and 0.14 approximately and respectively
# 
# *Conversion Rate*:  Ranges from 0.01 to 0.05 with a mean and std: of 0.25 and 0.14 approximately and respectively
# 
# *Bounce Rate*: Ranges from 0.2 to 0.8 with a mean and std: of 0.51 and 0.17 approximately and respectively
# 
# *Scroll_Depth*: Ranges from 20 to 80 with a mean and std: of 50.32 and 16.90 approximately and respectively
# 
# *Age*: The users Ranges from 18 to 65 years with a mean and std: of 41.5 and 14 approximately and respectively
# 
# *Session_Duration*: Users spend time which Ranges from 38 sec to 1797 sec(almost 30min) with a mean and std: of 925 secs(15 min) and 508 sec (8.5min) approximately and respectively

# In[102]:


data.info()
# 4 categorical features and 6 numerical features 
num_feat=data.select_dtypes(exclude="object")
cat_feat=data.select_dtypes(include="object")


# In[103]:


# grouping data by theme and then calculating the mean of each features 
# getting the theme rating based on each feature
theme_rating=data.groupby("Theme").mean()
print(theme_rating)
#sorting the theme rating by conversion for better analysis
theme_rating.sort_values("Conversion Rate",ascending=False)


# In[104]:


def cohen_d(data, col):

    # compute the mean of the themes per feature
    N1=data[data.Theme=='Light Theme'][col].mean()
    N2=data[data.Theme=='Dark Theme'][col].mean()

    # compute the standard deviation per feature
    SD1=data[data.Theme=='Light Theme'][col].std()
    SD2=data[data.Theme=='Dark Theme'][col].std()
    # compute the pooled standard deviation
    pooled_std=np.sqrt(((486-1)*(SD1**2) + (514-1)*(SD2**2))/(514+486-2))
    # compute the Cohen's d for the effect Size

    cohen_d=(N1-N2)/pooled_std
    return cohen_d

cohen_d={feat:np.round(cohen_d(data,feat),4) for feat in num_feat.columns}
cohen_d

INSIGHTS
1. ClickThroughRate: More users clicked a link/Button on the Dark Theme than Light Theme website by approximately 1% = DT
2. Conversion Rate: Larger percentage of users signed up after 1st visit on the Light Theme than Dark Theme by a margin of 1%.=LT
3. Bounce Rate: An averagely a user is bound to leave the site with Dark Theme as much as the one with Light theme because  the dark theme tend to be slightly on the high side by approximately 1% =LT
4. Scroll_Depth: Light theme has a higher Scroll depth than Dark Theme=LT
5. Age: There is no significant difference in the mean age of the users
6. Session Duration: the average session for a LT user(930.83 sec) is longer than that of Dark Theme users(919.48 sec)=LT

Conclusively and averagely, it is evident to say that the Light Theme design performed better than the Dark Theme design.  However, to further confirm this is why we are going to introduce Hypothesis Testing. The dark theme performed better only in terms of Click Through Rate while the Light Theme performed better in Conversion Rate,Bounce Rate, Scroll_depth, Session duration. However, there are no large difference between the metrics**HYPOTHESIS TESTING**
Here, we will consider using a significance level(alpha) of 0.05 and we will consider our experiment statistically sigificant if the the p-value is less than 0.05. This is the probability of rejecting the null hypothesis when it is true.

Null Hypothesis(Ho)= There is no significant difference between the Light and Dark Theme in terms of the metrics  
Alternative Hypothesis(Ha)= Significant difference does occur between the Light and Dark Theme in terms of the metrics



1. Click Through Rate(CTR): we choose to use two sample t-test to compare the means of two independent samples(Themes in this case)

# In[105]:


# getting the mean Click Through Rate for each theme
light_theme_ctr=data[data["Theme"]=="Light Theme"]["Click Through Rate"]
dark_theme_ctr=data[data["Theme"]=="Dark Theme"]["Click Through Rate"]

# hypothesis testing- 2 sample t-test
t_stat_ctr,p_val_ctr=ttest_ind(light_theme_ctr,dark_theme_ctr,equal_var=False)
print(f" For Click Through Rate : the t_stat_ctr= {t_stat_ctr} and p_val_ctr= {p_val_ctr}\n")

#   If True (default), perform a standard independent 2 sample test
#     that assumes equal population variances [1]_.
#     If False, perform Welch's t-test, which does not assume equal
#     population variance [2]_.

# getting the mean Conversion Rate for each theme
light_theme_cr=data[data["Theme"]=="Light Theme"]["Conversion Rate"]
dark_theme_cr=data[data["Theme"]=="Dark Theme"]["Conversion Rate"]

# hypothesis testing- 2 sample t-test
t_stat_cr,p_val_cr=ttest_ind(light_theme_cr,dark_theme_cr,equal_var=False)
print(f" For Conversion Rate : the t_stat_cr= {t_stat_cr} and p_val_cr= {p_val_cr}\n")

#   If True (default), perform a standard independent 2 sample test
#     that assumes equal population variances [1]_.
#     If False, perform Welch's t-test, which does not assume equal
#     population variance [2]_.


# getting the Bounce Rate for each theme
light_theme_br=data[data["Theme"]=="Light Theme"]["Bounce Rate"]
dark_theme_br=data[data["Theme"]=="Dark Theme"]["Bounce Rate"]

# hypothesis testing- 2 sample t-test
t_stat_br,p_val_br=ttest_ind(light_theme_br,dark_theme_br,equal_var=False)
print(f" For Bounce Rate : the t_stat_br= {t_stat_br} and p_val_br= {p_val_br}\n")


# getting the Scroll depth for each theme
light_theme_sd=data[data["Theme"]=="Light Theme"]["Scroll_Depth"]
dark_theme_sd=data[data["Theme"]=="Dark Theme"]["Scroll_Depth"]

# hypothesis testing- 2 sample t-test
t_stat_sd,p_val_sd=ttest_ind(light_theme_sd,dark_theme_sd,equal_var=False)
print(f" For  Scroll depth : the t_stat_sd= {t_stat_sd} and p_val_sd= {p_val_sd}\n")


# getting the Session_Duration for each theme
light_theme_sD=data[data["Theme"]=="Light Theme"]["Session_Duration"]
dark_theme_sD=data[data["Theme"]=="Dark Theme"]["Session_Duration"]

# hypothesis testing- 2 sample t-test
t_stat_sD,p_val_sD=ttest_ind(light_theme_sD,dark_theme_sD,equal_var=False)
print(f" For Session_Duration': the t_stat_sD= {t_stat_sD} and p_val_sD= {p_val_sD}")


# **INSIGHTS**
# 1. Click Through Rate : the p-value < 0.05 thus there is no reason to accept the null hypothesis and for this reason there is a statistically significant difference in click through rate between the themes based on the data provided. The dark theme tends to have more clicks than the light theme. Ha is true
# 
# 
# 2. Conversion Rate : the p-value > 0.05 thus there is no reason to reject the null hypothesis and for this reason there is no statistically significant difference in conversion rate for both themes based on the data provided. Ho is true
# 
# 
# 3. Bounce Rate : the p-value > 0.05 thus there is no reason to reject the null hypothesis and for this reason there is no statistically significant difference in Bounce Rate between the themes based on the data provided. Ho is true
# 
# 
# 4. Scroll depth : the p-value > 0.05 thus there is no reason to reject the null hypothesis and for this reason there is no statistically significant difference in Scroll depth between the themes based on the data provided. Ho is true
# 
# 
# 5.  Session_Duration : the p-value > 0.05 thus there is no reason to reject the null hypothesis and for this reason there is no statistically significant difference in Session_Duration between the themes based on the data provided. Ho is true
# 
# 
# 
# Based on the hypothesis testing above, the Click Through Rate is the only metrics that showed that the Dark Theme will attract more user engagements that the Light Theme. Other metrics measures that the user engagement and interaction is not affected by the choice of theme based on the data.
# 
# 

# In[106]:


def hypothesis_testing_theme(df,col):
    #getting the features value in the data
    light_theme=df[df["Theme"]=="Light Theme"][col]
    dark_theme=df[df["Theme"]=="Dark Theme"][col]
    # hypothesis testing- 2 sample t-test
    t_stat_,p_val_=ttest_ind(light_theme,dark_theme,equal_var=False)
    return t_stat_,p_val_
for i in num_feat:
    print(f"For {i}, the t_stat and p_value are :  {hypothesis_testing_theme(data,i)}\n")


# In[107]:


comparison_table=pd.DataFrame({"Metrics": ['Click Through Rate','Conversion Rate','Bounce Rate','Scroll_Depth','Session_Duration'],
                               "T-statistics": [t_stat_ctr,t_stat_cr,t_stat_br,t_stat_sd,t_stat_sD],
                               "P-value": [p_val_ctr,p_val_cr,p_val_br,p_val_sd,p_val_sD]
                              })
print("The result of The Statistical Testing(two sample T-Test) based on significance level, alpha=0.05")
comparison_table


# NOTES FROM STATISTICAL TEST METHOD - one Sample T- test, 2 sample T-test, Paired T-test, Type 1 and Type 2 error 
# 
# References:
# Hamelg. (2015, November 20). Python for Data Analysis Part 24: Hypothesis testing and the T-test. Life Is Study. https://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-24.html 
# 
# >> Hypothesis Testing makes it possible to determine if an observed data deviates from what it expected. The Scipy.stats in python has a lot functions that makes this possible to do.
# 
# >> This test is based on a statistical test named null hypothesis which expresses that nothing is likely to happen when comparing two samples/variables. For instance, you have 2 recipes for making an apple pie,you want to make a decision on which recipe produces the apple pie with the best taste or you want to compare the average age of voters in a population(UK) with average age of voters in Wales. Your null hypothesis [Ho]. says that the 2 recipes or the average age of voters in each sample are the same, meaning no effect, no stance. On the other hand, if there exist any statistically significant difference, we reject the null hypothesis and accept the alternative hypothesis [Ha].
# 
# >> Next is to set the significance level, α, after identifying the Ho and Ha, which is the probability threshold to determine whether to reject the null hypothesis. Then we choose the T-test (for numerical data sample) to use depending on the no of groups/samples we want to compare. From the t-test, we can compute the p-value which is the main probability of rejecting the null hypothesis when it is true. If the p-value(the probability of getting result as less or more extreme than the observed value) is less than the α stated, we reject the null hypothesis in favour of the alternative hypothesis. This means that there is statistically significant difference.
#                    - Significance level, α = 1 - confidence level
# >> T-Test is the statistical test that computes the p-value and a determinant for accepting or rejecting a null hypothesis and concluding whether a sample differs from one another.
#   
#    - One Sample T - test : This is performed to determine if a sample mean differs from the whole population mean e.g if the population mean of voters in wales differs from the UK population mean. The function *stats.ttest_1samp()* makes this possible and it also gives the t-statistics and p-value. T -statistics measures how the sample mean deviates from the [Ho]
#   
#    - Two Sample T - test : This has been applied for comparing means of two independent groups or samples using the function 
#       *stats.ttest_ind()
#   
#    - Paired T- test  : This is used to find the statistically signifcant difference that exist between samples in the same group. e.g. weight loss difference before and after using a drug. This is possible in python using the function, *stats.ttest_rel()*
#    
# 
# 
# >>> Type I and Type II error: Since result of hypothesis test and the decision to accept or reject a null hypothesis after test is not 100%. The possibility of making incorrect conclusion from hypothesis testing is divided into: 
#            - TYPE I error / False Positive: This is when you reject the null Hypothesis when it is actually true. It is characterized by lower significance level which reduces the risk of False Positive or Type 1 error and increases that of Type 2 error 
#            - TYPE II error/ False Negative: This is when you accept the null Hypothesis when it is actually false.It is characterized with higher significance level which increases the chance of a Type I but reduces the risk of a Type II error
# 
# 
# Significance level, alpha is a threshold that helps set the balance between making decisions and avoiding errors:
# For example, let's say you want to find out likelihood of someone contracting covid within a post code, a LOWER ALPHA reduces the chance of Type I error /False Positive(which is incorrectly concluding that someone has covid when they dont) and increases the chance of Type II/False Negative(which is incorrectly concluding that someone does not have when they have) - In this case Type II is the worst error to make. This is vice versa for a HIGHER ALPHA. 
# 
# - Therefore for the example above, a lower alpha makes it difficult to find someone with covid and higher alpha makes detection easier. 
# - In addition, if p-value < alpha - you reject the null hypothesis and accept the alternative hypothesis
# -              if  p-value > alpha - you accept the null hypothesis and reject the alternative hypothesis

# In[108]:


#A/B TESTING USING PYTHON FOR THE TWO THEMES USING Z - STATISTICS
data.head()

#checking the correlation between numerical data
data.corr() # no profound correlation for the combined data


# In[109]:


# Since the conversion rate measures the amount of sessions ending up in transactions, 
# and the CTR measures the proportion of users interacting with the website. lets observe the realtionshipe that
# occus between the two

import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# separate the light and dark them data 
light_theme_data=data[data['Theme']=='Light Theme']
dark_theme_data=data[data['Theme']=='Dark Theme']
fig=px.scatter(data_frame=data,
               x='Click Through Rate',
               y='Conversion Rate',
               color='Theme',trendline='ols',opacity=1.0)

fig.update_layout(title_text='Figure 1 : Conversion Rate Vs Click Through Rate')

fig.show()

The relationship between the CTR and CR shows a consistent trend. The distribution of both metrics for each theme is widespread and normal.Thus as we have more users interacting with the website, so we have more users signing up at first visit. In adition, with both themes, a user could end up signing up regardless of the amount of clicks made.
# In[110]:


# distribution of the ClickThrough Rate Using histogram 
fig=go.Figure()

#create a grouped histogram plot on a figure
fig.add_trace(go.Histogram(x=light_theme_data['Click Through Rate'],name='light theme',opacity=1.0))
fig.add_trace(go.Histogram(x=dark_theme_data['Click Through Rate'],name='dark theme',opacity=1.0))

fig.update_layout(title_text='Figure 2 : Click Through Rate For Each Theme',
                 xaxis_title_text='Click Through Rate',
                 yaxis_title_text='Frequency',
                 barmode='group',
                 bargap=0.15)
fig.show()

From figure 2 the click through rate distribution appears to be normal for both theme and the difference observed is insignificant. However the dark theme appears to have a slightly high CTR than the light theme.
# In[111]:


# distribution of the Conversion Rate Using histogram 
fig=go.Figure()

#create a grouped histogram plot on a figure
fig.add_trace(go.Histogram(x=light_theme_data['Conversion Rate'],name='light theme',opacity=1.0))
fig.add_trace(go.Histogram(x=dark_theme_data['Conversion Rate'],name='dark theme',opacity=1.0))

# add the labels 
fig.update_layout(title_text='Figure 3 : Conversion Rate For Each Theme',
                 xaxis_title_text='Conversion Rate',
                 yaxis_title_text='Frequency',
                 barmode='group',
                 bargap=0.15)
fig.show()

It appears that conversion rate for both themes are normally distributed and there is no significant difference in performance as observed above from Figure 3. However the dark theme appears to have a slightly higher CR than the light theme.
# In[112]:


# Comparing the Scroll Depth Rate to determine the difference in performance that will bring about improvement.

# distribution of the Scroll Depth Using Box Plot 
fig=go.Figure()

#create a 2 Box plot on a figure
fig.add_trace(go.Box(x=light_theme_data['Scroll_Depth'],name='light theme',opacity=1.0))
fig.add_trace(go.Box(x=dark_theme_data['Scroll_Depth'],name='dark theme',opacity=1.0))

# add the labels 
fig.update_layout(title_text='Figure 4 : Scroll_Depth For Each Theme',
                 yaxis_title_text='Scroll Depth')
fig.show()


# Comparing the Bounce Rate to determine the difference in performance that will bring about improvement.

# distribution of the Bounce Rate Using Box Plot 
fig=go.Figure()

#create a 2 Box plot on a figure
fig.add_trace(go.Box(x=light_theme_data['Bounce Rate'],name='light theme',opacity=1.0))
fig.add_trace(go.Box(x=dark_theme_data['Bounce Rate'],name='dark theme',opacity=1.0))

# add the labels 
fig.update_layout(title_text='Figure 5 : Bounce Rate For Each Theme',
                 yaxis_title_text='Bounce Rate')
fig.show()

From Figure 4 & 5, it could be observed that users tend to hoover more around features on Light Theme and will not likely leave the website immediately after seeing a single page compared to the Dark Theme. Whilst the difference in performance of both theme as per Bounce Rate and Scroll Depth rate is not much;
From the box plot, the median scroll depth for the Light(51.4995) is slightly higher than the Dark Theme(50.0173)
From the box plot, the median Bounce Rate for the Light(0.4968) is slightly lower than the Dark Theme(0.5322)
# In[113]:


data.columns


# In[ ]:





# In[114]:


# Comparing the Average Session Duration for both Theme, 
# Here we use two sample t -test  
avg_light_theme= data[data['Theme']=='Light Theme'].Session_Duration.mean()
avg_dark_theme= data[data['Theme']=='Dark Theme'].Session_Duration.mean()

from scipy.stats import ttest_ind
# z test for both themes session duration of users 
tstat,pval=stats.ttest_ind( data[data['Theme']=='Light Theme'].Session_Duration,data[data['Theme']=='Dark Theme'].Session_Duration)

print('The average session duration for light theme: ',round(avg_light_theme,2), 'sec')
print('The average session duration for Dark theme: ',round(avg_dark_theme,2), 'sec')
print(' Z -Statistics : ', tstat)
print(' P- Value : ', pval)


# In[115]:


# Comparing the Purchases made on both themes 
# we use z statistics as we are sure of the variances and the sample > 30
#convesion rate = [No of Purchases/Total No of Samples]

# separate the light and dark them data 
light_theme_data=data[data['Theme']=='Light Theme']
dark_theme_data=data[data['Theme']=='Dark Theme']



light_theme_purchases= light_theme_data[light_theme_data.Purchases=='Yes'].shape[0]
dark_theme_purchases= dark_theme_data[dark_theme_data.Purchases=='Yes'].shape[0]

# conversion rate for each theme 
light_conversion= light_theme_purchases/light_theme_data.shape[0]
dark_conversion=dark_theme_purchases/dark_theme_data.shape[0]

# compute the sample size and conversion counts
sample_size=[light_theme_data.shape[0],dark_theme_data.shape[0]]
conversion_counts=[light_theme_purchases,dark_theme_purchases]
#proportions_ztest
#compute z statistic based on  conversion rate 
zstat,pval=proportions_ztest(conversion_counts,sample_size)


print('The average conversion_rates based on purchases for light theme: ',light_conversion)
print('The average conversion_rates based on purchases for Dark theme: ',dark_conversion)
print(' Z -Statistics : ', zstat)
print(' P- Value : ', pval)

# The average session duration for light theme:  930.83 sec
# The average session duration for Dark theme:  919.48 sec
#  Z -Statistics :  0.3528
#  P- Value :  0.7243

The z-statistics is a measure of the observed difference between both theme performance (based on purchases and session duration) in terms of standard deviation. The positive z-statistics shows that the light theme has higher conversion rates than the dark theme. 

The p-values for both metrics are greater than the commonly used significance level(alpha=0.05) for A/B testing. This means that the probability of accepting the null hypothesis(no statistically significant difference) is greater than the significance level. Thus there are no statistically significant difference between the purchases made and the session duration of both themes and any observed difference is due to random variation in data. Simply, based on data and analysis, there are no significant difference between both themes in  terms of purchases and session duration. 
 
# The average conversion_rates based on purchases for light theme:  0.5308
# The average conversion_rates based on purchases for Dark theme:  0.5039
#  Z -Statistics :  0.8531
#  P- Value :  0.3938
# In[116]:


#- Why Z test and t-test , t-statistics and z statistics 

