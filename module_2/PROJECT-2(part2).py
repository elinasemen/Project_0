#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

pd.set_option('display.max_rows', 50) # показывать больше строк
pd.set_option('display.max_columns', 50) # показывать больше колонок

students = pd.read_csv('stud_math.csv')


# In[2]:


display(students.head(10))
students.info()


# ## Блок с функциями и полезностями

# In[3]:


# Создадим два списка с числовыми и номинативными значениями
object_col = students.select_dtypes(include=['object']).columns
float_col = students.select_dtypes(include=['int64', 'float64']).columns


# In[4]:


#scatter plot
def scatter_plot (column):
    var = column
    data = pd.concat([students['score'], students[var]], axis=1)
    data.plot.scatter(x=var, y='score');


# In[5]:


#distribution plot
def dist_plot(column, p_bin = 20):
    fig = plt.figure()
    axes = fig.add_axes([0,0,1,1])
    axes.hist(students[column], bins = p_bin, color='grey')
    axes.set_ylabel('Количество '+column)
    axes.set_xlabel(column)


# In[6]:


def dist_iqr(column):
    IQR = students[column].quantile(0.75) - students[column].quantile(0.25)
    perc25 = students[column].quantile(0.25)
    perc75 = students[column].quantile(0.75)
    print('25-й перцентиль: {},'.format(perc25),
          '75-й перцентиль: {},'.format(perc75),
          "IQR: {}, ".format(IQR),
          "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
    
    students[column].loc[students[column].between(
        perc25 - 1.5*IQR,
        perc75 + 1.5*IQR)].hist(bins = 16, label = 'IQR')
    
    plt.legend();


# In[7]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.boxplot(x=column, y='score', 
                data=students.loc[students.loc[:, column].isin(students.loc[:, column].value_counts().index[:10])],
               ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[61]:


def get_countplot(column):
    fig, ax = plt.subplots(figsize = (14, 4))
    sns.countplot(x=column,
                  data=students, 
                  ax = ax)
    ax.set_title('Countplot for ' + column)
    plt.show()


# In[9]:


def get_stat_dif(column):
    cols = students.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(students.loc[students.loc[:, column] == comb[0], 'score'], 
                     students.loc[students.loc[:, column] == comb[1], 'score']).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# ## EDA

# Исследуем целевую переменную

# In[10]:


students.score.describe()


# In[11]:


sns.distplot(students.score)


# Можно заметить, что есть некоторое количество учеников, получивших 0 за экзамен. Исключать наблюдения, равные 0, зависит от контекста: можно ли было набрать 0 баллов или нет.

# Исследуем зависимости целевой переменной от числовых признаков.

# In[12]:


for col in float_col:
    scatter_plot(col)


# Можно заметить кое-какие выбросы для некоторых переменных, далее рассмотрим подробнее их распределения. По данным же графикам пока сказать нечего, за исключением того, что все числовые переменные дискретны и более-менее равномерно распределены среди значений целевой переменной.

# In[13]:


float_col


# AGE (возраст ученика)

# In[14]:


dist_plot('age', 15)


# По графику выше можно заметить резкое уменьшение количества детей в возрасте от 20 лет. Неизвестно, может ли ученику быть более 19 лет, я бы предположила, что нет.
# Далее попробуем определить выбросы по формуле межквартильного размаха.

# In[15]:


dist_iqr('age')


# Согласно графика выше, выбросом являются записи об учениках в возрасте 22 года. Можно удалить эти записи. Также можно будет удалить записи об учениках в возрасте 20-21 год, после получения соответствуюшей информации. 

# In[16]:


students = students.drop(students[students.age == 22].index)


# Medu (образование отца)

# In[17]:


students.Medu.describe()


# In[18]:


students[students['Medu'].isnull() == True]


# In[19]:


dist_plot('Medu')


# In[20]:


dist_iqr('Medu')


# Мы видим отсутствующие данные по трем ученикам. Можно заполнить их медианным значением равным 3.

# In[21]:


students = students.fillna({'Medu':3})


# Fedu (образование матери)

# In[22]:


students.Fedu.describe()


# In[23]:


students[students['Fedu'].isnull() == True]


# In[24]:


dist_plot('Fedu')


# В случае с уровнем образования матери мы наблюдаем явный выброс со значением 40. Предположить, что должно было быть записано, мы не можем, проще удалить эту запись. 
# Пропуски в количестве 24 штук можно заполнить медианным значением = 2.

# In[25]:


students = students.drop(students[students.Fedu == 40].index)


# In[26]:


students = students.fillna({'Fedu':2})


# traveltime (время в пути до школы)

# In[27]:


students.traveltime.describe()


# In[28]:


dist_plot('traveltime')


# In[29]:


dist_iqr('traveltime')


# Несмотря на то, что время дороги до школы, равное 4 часам, вышло за границы выбросов, удалять мы такие записи не будем.
# Пропуски заполним медианным значением, равным 1.

# In[30]:


students = students.fillna({'traveltime':1})


# studytime (время на учёбу помимо школы в неделю)

# In[31]:


students.studytime.describe()


# In[32]:


dist_plot('studytime')


# In[33]:


#пропуски медианой
students = students.fillna({'studytime':2})


# failures (количество внеучебных неудач)

# In[34]:


students.failures.describe()


# In[35]:


dist_plot('failures')


# In[36]:


#Заполнение пропусков медианой
students = students.fillna({'failures':0})


# Так как количество наблюдений со значением failures = 0  гораздо больше прочих, возможно, имеет смысл объединить значения 1<=n<=3, например, в 1. 

# famrel (семейные отношения)

# In[37]:


students.famrel.describe()


# In[38]:


dist_plot('famrel')


# In[39]:


#удаляем значение -1
students = students.drop(students[students.famrel == -1].index)


# In[40]:


#пропуски медианой
students = students.fillna({'famrel':4})


# freetime (свободное время после школы)

# In[41]:


students.freetime.describe()


# In[42]:


dist_plot('freetime')


# In[43]:


#пропуски медианой
students = students.fillna({'freetime':3})


# goout (проведение времени с друзьями)

# In[44]:


students.goout.describe()


# In[45]:


dist_plot('goout')


# In[46]:


#пропуски медианой
students = students.fillna({'goout':3})


# health (текущее состояние здоровья)

# In[47]:


students.health.describe()


# In[48]:


dist_plot('health')


# In[49]:


#пропуски медианой
students = students.fillna({'health':4})


# absences (количество пропущенных занятий)

# In[50]:


students.absences.describe()


# In[51]:


dist_plot('absences')


# In[52]:


dist_iqr('absences')


# Удалим все то, что выходит за границы выбросов.

# In[53]:


students = students.drop(students[students.absences >= 20].index)


# Далее проведем анализ номинативных переменных

# In[54]:


# Посмотрим распределение номинативных переменных
for col in object_col:
    get_countplot(col)


# Во-первых, у большинства (13 из 17) номинативных переменных всего 2 уникальных значения.
# Рассмотрим каждую переменную отдельно.
# 
# Большинство учеников:
#  1. Посещали школу MS.
#  2. Живут с родителями.
#  3. С опекуном - матерью.
#  4. Не получали дополнительную образовательную поддержку.
#  5. Посещали детский сад.
#  6. Хотят получить высшее образование.
#  7. Имеют доступ к интернету. 
# 
# По остальным признакам ничего примечательного, распределены более-менее равномерно.
# 
# Далее оценим степень влияния на целевую переменную номинативных переменных. 

# In[55]:


for col in object_col:
    get_boxplot(col)


# По данным графикам можно предположить, что влияние на целевую переменную оказывают такие показатели, как Mjob, Fjob, schoolsup, higher. Но в последних двух показателях был заметный перекос в сторону одного из значений, поэтому использовать их в построении модели чревато переобучением. 
# 
# Далее попробуем определить статистическую значимость показателей. Я пока не сильна в статистике и проверке гипотез, поэтому недолго думая беру формулу из предыдущего EDA про шоколадные батончики. 

# In[56]:


for col in object_col:
    get_stat_dif(col)


# Проверка на статистическую значимость показала, что значимым является только номинативный показатель Mjob. Таким образом, из номинативных переменных можно оставить только переменную Mjob. 

# In[58]:


for col in object_col:
    if col != 'Mjob':
        students.drop(col, axis=1, inplace=True)


# In[60]:


# Заменим пропуски в колонке Mjob значениями others
students = students.fillna({'Mjob':'others'})


# Далее проведем корреляционный анализ для числовых переменных

# In[62]:


correlation = students.corr()


# In[66]:


print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# Сразу можно выделить сильную обратную зависимость между переменными studytime и studytime, granular. 
# Последнюю можно удалить.

# In[67]:


students.drop('studytime, granular', axis=1, inplace=True)


# In[70]:


correlation = students.corr()
# print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# Следующая зависимость, котрую мы наблюдаем, это зависимость между переменными Fedu, Medu. Так как Medu сильнее влияет на score, удалим Fedu.

# In[71]:


students.drop('Fedu', axis=1, inplace=True)


# In[72]:


correlation = students.corr()
# print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm')


# Если проанализировать окончательную матрицу корреляций, средм значимых переменных можно выделить age, Medu, studytime, failures, goout. 

# In[73]:


for col in ['traveltime', 'famrel', 'freetime', 'health', 'absences']:
    students.drop(col, axis=1, inplace=True)


# In[76]:


students.columns


# ## Вывод

# В ходе проведенного анализа я попрактиковалась в проверке качества данных и их очистке и обработке, сформулировала гипотезы касаемо влияния переменных на целевую, проверила эти гипотезы с использованием инструментов математической статистики. 
# 
# Сами данные были очищены от выбросов (в переменных age, Fedu, famrel, absences), заполнены значения Null значениями медианы для данного признака. Медиана выбиралась, так как средняя не являлась целым числом, а все числовые переменные дискретны.
# 
# В качестве входных параметров будущей модели были выбраны age, Medu, Mjob, studytime, failures, goout. Но на самом деле, при построении реальной модели, лучше убирать переменные постепенно и проверять, как при этом меняется качество модели. 
