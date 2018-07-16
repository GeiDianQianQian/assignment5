import sys
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
from scipy import stats



OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)

def getYear(date):
    year = date.year
    return year

def T_test(weekday_counts, weekend_counts):
    weekday_normal = stats.normaltest(weekday_counts['comment_count'])
    weekend_normal = stats.normaltest(weekend_counts['comment_count'])
    levene = stats.levene(weekday_counts['comment_count'], weekend_counts['comment_count'])
    test = stats.ttest_ind(weekday_counts['comment_count'], weekend_counts['comment_count'])
    return weekday_normal, weekend_normal, levene, test

def get_log(counts):
    return (np.log(counts))

def get_exp(counts):
    return (np.exp(counts))

def get_sqrt(counts):
    return (np.sqrt(counts))

def get_times(counts):
    return(counts**2)

def transform(weekday_counts, weekend_counts): #get_sqrt is the best, but still not passed
    weekday_copy = weekday_counts.copy()
    weekend_copy = weekend_counts.copy()
    weekday_copy['comment_count'] = weekday_counts['comment_count'].apply(get_sqrt)
    weekend_copy['comment_count'] = weekend_counts['comment_count'].apply(get_sqrt)
    weekday_transform = stats.normaltest(weekday_copy['comment_count'])
    weekend_transform = stats.normaltest(weekend_copy['comment_count'])
    levene_transform = stats.levene(weekday_copy['comment_count'], weekend_copy['comment_count'])
    #test_transform = stats.ttest_ind(weekday_transform['comment_count'], weekend_transform['comment_count'])
    return weekday_transform, weekend_transform, levene_transform

def get_isoYear(date):
    isoYear = date.isocalendar()[0]
    return isoYear

def get_isoWeek(date):
    isoWeek = date.isocalendar()[1]
    return isoWeek

def centralLimit(counts):
    counts_copy = counts
    counts_copy['isoYear'] = counts['date'].apply(get_isoYear)
    counts_copy['isoWeek'] = counts['date'].apply(get_isoWeek)
    weekday_centralLimit = counts_copy[(counts_copy['week'] == 'weekday')]
    weekend_centralLimit = counts_copy[(counts_copy['week'] == 'weekend')]
    weekday_mean = weekday_centralLimit.groupby(['isoYear', 'isoWeek']).agg('mean')
    weekend_mean = weekend_centralLimit.groupby(['isoYear', 'isoWeek']).agg('mean')
    return weekday_mean,weekend_mean

def main():
    f = open('1.csv')
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
    #print(counts)
    #separate year, month and day
    counts['year'] = counts['date'].apply(getYear)
    counts = counts[(counts['year'] == 2012) | (counts['year'] == 2013)]
    #print(counts)
    #choose only canada
    counts = counts[(counts['subreddit'] == 'canada')]
    #print(counts)

    #separate weekday and weekend
    #reference: https://stackoverflow.com/questions/46129799/count-workdays-vs-weekends-usage-in-pandas
    counts['date'] = pd.to_datetime(counts['date'], infer_datetime_format=True)
    counts['week'] = np.where(counts['date'].dt.dayofweek < 5, 'weekday', 'weekend')
    #print(counts)
    weekday_counts = counts[(counts['week'] == 'weekday')]
    weekend_counts = counts[(counts['week'] == 'weekend')]
    #print(weekday_counts)
    #print(weekend_counts)

    # do T-test
    weekday_normal, weekend_normal, levene, test = T_test(weekday_counts, weekend_counts)
    #print(weekday_normal)
    #print(weekend_normal)
    #print(levene)
    #print(test)

    #fix 1
    weekday_transform, weekend_transform, levene_transform = transform(weekday_counts, weekend_counts)
    print (weekday_transform)
    #print (weekend_transform)

    #fix 2
    weekday_mean,weekend_mean = centralLimit(counts)
    weekday_weekly,weekend_weekly, levene_weekly, test_weekly =T_test(weekday_mean,weekend_mean)

    #fix 3
    utest = stats.mannwhitneyu(weekday_counts['comment_count'], weekend_counts['comment_count'])

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p = test.pvalue,
        initial_weekday_normality_p = weekday_normal.pvalue,
        initial_weekend_normality_p = weekend_normal.pvalue,
        initial_levene_p = levene.pvalue,
        transformed_weekday_normality_p = weekday_transform.pvalue,
        transformed_weekend_normality_p = weekend_transform.pvalue,
        transformed_levene_p = levene_transform.pvalue,
        weekly_weekday_normality_p = weekday_weekly.pvalue,
        weekly_weekend_normality_p = weekend_weekly.pvalue,
        weekly_levene_p = levene_weekly.pvalue,
        weekly_ttest_p = test_weekly.pvalue,
        utest_p = utest.pvalue,
    ))


if __name__ == '__main__':
    main()
