import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def loaddata(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "loaddata",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    path = "data/"

    PREDICTION_LABEL = 'Survived'

    test_df = pd.read_csv(path + "test.csv")
    train_df = pd.read_csv(path + "train.csv")

    # -----------------------DATA SAVING START---------------------------------
    if "test_df" in locals():
        _kale_resource_save(test_df, os.path.join(
            _kale_data_directory, "test_df"))
    else:
        print("_kale_resource_save: `test_df` not found.")
    if "PREDICTION_LABEL" in locals():
        _kale_resource_save(PREDICTION_LABEL, os.path.join(
            _kale_data_directory, "PREDICTION_LABEL"))
    else:
        print("_kale_resource_save: `PREDICTION_LABEL` not found.")
    if "train_df" in locals():
        _kale_resource_save(train_df, os.path.join(
            _kale_data_directory, "train_df"))
    else:
        print("_kale_resource_save: `train_df` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def datapreprocessing(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "datapreprocessing",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "test_df" not in _kale_directory_file_names:
        raise ValueError("test_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "test_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "test_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    test_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    data = [train_df, test_df]
    for dataset in data:
        dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
        dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
        dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
        dataset['not_alone'] = dataset['not_alone'].astype(int)
    train_df['not_alone'].value_counts()
    # This does not contribute to a person survival probability
    train_df = train_df.drop(['PassengerId'], axis=1)
    import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [train_df, test_df]

    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(
            lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)
    # we can now drop the cabin feature
    train_df = train_df.drop(['Cabin'], axis=1)
    test_df = test_df.drop(['Cabin'], axis=1)
    data = [train_df, test_df]

    for dataset in data:
        mean = train_df["Age"].mean()
        std = test_df["Age"].std()
        is_null = dataset["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        # fill NaN values in Age column with random values generated
        age_slice = dataset["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        dataset["Age"] = age_slice
        dataset["Age"] = train_df["Age"].astype(int)
    train_df["Age"].isnull().sum()
    train_df['Embarked'].describe()
    # fill with most common value
    common_value = 'S'
    data = [train_df, test_df]

    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
    train_df.info()

    # -----------------------DATA SAVING START---------------------------------
    if "test_df" in locals():
        _kale_resource_save(test_df, os.path.join(
            _kale_data_directory, "test_df"))
    else:
        print("_kale_resource_save: `test_df` not found.")
    if "train_df" in locals():
        _kale_resource_save(train_df, os.path.join(
            _kale_data_directory, "train_df"))
    else:
        print("_kale_resource_save: `train_df` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def featureengineering(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "featureengineering",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "PREDICTION_LABEL" not in _kale_directory_file_names:
        raise ValueError("PREDICTION_LABEL" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "PREDICTION_LABEL"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "PREDICTION_LABEL" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    PREDICTION_LABEL = _kale_resource_load(
        os.path.join(_kale_data_directory, _kale_load_file_name))

    if "test_df" not in _kale_directory_file_names:
        raise ValueError("test_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "test_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "test_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    test_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    data = [train_df, test_df]

    for dataset in data:
        dataset['Fare'] = dataset['Fare'].fillna(0)
        dataset['Fare'] = dataset['Fare'].astype(int)
    data = [train_df, test_df]
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(
            ' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)
    train_df = train_df.drop(['Name'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    genders = {"male": 0, "female": 1}
    data = [train_df, test_df]

    for dataset in data:
        dataset['Sex'] = dataset['Sex'].map(genders)
    train_df = train_df.drop(['Ticket'], axis=1)
    test_df = test_df.drop(['Ticket'], axis=1)
    ports = {"S": 0, "C": 1, "Q": 2}
    data = [train_df, test_df]

    for dataset in data:
        dataset['Embarked'] = dataset['Embarked'].map(ports)
    data = [train_df, test_df]
    for dataset in data:
        dataset['Age'] = dataset['Age'].astype(int)
        dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
        dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
        dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
        dataset.loc[dataset['Age'] > 66, 'Age'] = 6

    # let's see how it's distributed train_df['Age'].value_counts()
    data = [train_df, test_df]

    for dataset in data:
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (
            dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (
            dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[(dataset['Fare'] > 31) & (
            dataset['Fare'] <= 99), 'Fare'] = 3
        dataset.loc[(dataset['Fare'] > 99) & (
            dataset['Fare'] <= 250), 'Fare'] = 4
        dataset.loc[dataset['Fare'] > 250, 'Fare'] = 5
        dataset['Fare'] = dataset['Fare'].astype(int)
    data = [train_df, test_df]
    for dataset in data:
        dataset['Age_Class'] = dataset['Age'] * dataset['Pclass']
    for dataset in data:
        dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
        dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
    # Let's take a last look at the training set, before we start training the models.
    train_df.head(10)
    train_labels = train_df[PREDICTION_LABEL]
    train_df = train_df.drop(PREDICTION_LABEL, axis=1)

    # -----------------------DATA SAVING START---------------------------------
    if "train_labels" in locals():
        _kale_resource_save(train_labels, os.path.join(
            _kale_data_directory, "train_labels"))
    else:
        print("_kale_resource_save: `train_labels` not found.")
    if "train_df" in locals():
        _kale_resource_save(train_df, os.path.join(
            _kale_data_directory, "train_df"))
    else:
        print("_kale_resource_save: `train_df` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def decisiontree(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "decisiontree",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "train_labels" not in _kale_directory_file_names:
        raise ValueError("train_labels" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_labels"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_labels" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_labels = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_df, train_labels)
    acc_decision_tree = round(decision_tree.score(
        train_df, train_labels) * 100, 2)

    # -----------------------DATA SAVING START---------------------------------
    if "acc_decision_tree" in locals():
        _kale_resource_save(acc_decision_tree, os.path.join(
            _kale_data_directory, "acc_decision_tree"))
    else:
        print("_kale_resource_save: `acc_decision_tree` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def svm(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "svm",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "train_labels" not in _kale_directory_file_names:
        raise ValueError("train_labels" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_labels"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_labels" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_labels = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    linear_svc = SVC(gamma='auto')
    linear_svc.fit(train_df, train_labels)
    acc_linear_svc = round(linear_svc.score(train_df, train_labels) * 100, 2)

    # -----------------------DATA SAVING START---------------------------------
    if "acc_linear_svc" in locals():
        _kale_resource_save(acc_linear_svc, os.path.join(
            _kale_data_directory, "acc_linear_svc"))
    else:
        print("_kale_resource_save: `acc_linear_svc` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def naivebayes(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "naivebayes",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "train_labels" not in _kale_directory_file_names:
        raise ValueError("train_labels" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_labels"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_labels" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_labels = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    gaussian = GaussianNB()
    gaussian.fit(train_df, train_labels)
    acc_gaussian = round(gaussian.score(train_df, train_labels) * 100, 2)

    # -----------------------DATA SAVING START---------------------------------
    if "acc_gaussian" in locals():
        _kale_resource_save(acc_gaussian, os.path.join(
            _kale_data_directory, "acc_gaussian"))
    else:
        print("_kale_resource_save: `acc_gaussian` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def logisticregression(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "logisticregression",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "train_labels" not in _kale_directory_file_names:
        raise ValueError("train_labels" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_labels"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_labels" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_labels = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    logreg = LogisticRegression(solver='lbfgs', max_iter=110)
    logreg.fit(train_df, train_labels)
    acc_log = round(logreg.score(train_df, train_labels) * 100, 2)

    # -----------------------DATA SAVING START---------------------------------
    if "acc_log" in locals():
        _kale_resource_save(acc_log, os.path.join(
            _kale_data_directory, "acc_log"))
    else:
        print("_kale_resource_save: `acc_log` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def randomforest(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "randomforest",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "train_labels" not in _kale_directory_file_names:
        raise ValueError("train_labels" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_labels"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_labels" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_labels = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "train_df" not in _kale_directory_file_names:
        raise ValueError("train_df" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "train_df"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "train_df" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    train_df = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(train_df, train_labels)
    acc_random_forest = round(random_forest.score(
        train_df, train_labels) * 100, 2)

    # -----------------------DATA SAVING START---------------------------------
    if "acc_random_forest" in locals():
        _kale_resource_save(acc_random_forest, os.path.join(
            _kale_data_directory, "acc_random_forest"))
    else:
        print("_kale_resource_save: `acc_random_forest` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def results(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "results",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "acc_log" not in _kale_directory_file_names:
        raise ValueError("acc_log" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "acc_log"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "acc_log" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    acc_log = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "acc_random_forest" not in _kale_directory_file_names:
        raise ValueError("acc_random_forest" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "acc_random_forest"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "acc_random_forest" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    acc_random_forest = _kale_resource_load(
        os.path.join(_kale_data_directory, _kale_load_file_name))

    if "acc_decision_tree" not in _kale_directory_file_names:
        raise ValueError("acc_decision_tree" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "acc_decision_tree"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "acc_decision_tree" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    acc_decision_tree = _kale_resource_load(
        os.path.join(_kale_data_directory, _kale_load_file_name))

    if "acc_gaussian" not in _kale_directory_file_names:
        raise ValueError("acc_gaussian" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "acc_gaussian"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "acc_gaussian" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    acc_gaussian = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "acc_linear_svc" not in _kale_directory_file_names:
        raise ValueError("acc_linear_svc" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "acc_linear_svc"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "acc_linear_svc" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    acc_linear_svc = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from matplotlib import style

    from sklearn import linear_model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    results = pd.DataFrame({
        'Model': ['Support Vector Machines', 'logistic Regression',
                  'Random Forest', 'Naive Bayes', 'Decision Tree'],
        'Score': [acc_linear_svc, acc_log,
                  acc_random_forest, acc_gaussian, acc_decision_tree]})
    result_df = results.sort_values(by='Score', ascending=False)
    result_df = result_df.set_index('Score')
    print(result_df)


def final_auto_snapshot(rok_workspace_aidays01_2rlcyd0k8_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/examples/titanic-ml-dataset/.titanic_dataset_ml.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("titanic-ml-fylgn",
                                     "final_auto_snapshot",
                                     "/home/jovyan/examples/titanic-ml-dataset/titanic_dataset_ml.ipynb")


loaddata_op = comp.func_to_container_op(
    loaddata, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


datapreprocessing_op = comp.func_to_container_op(
    datapreprocessing, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


featureengineering_op = comp.func_to_container_op(
    featureengineering, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


decisiontree_op = comp.func_to_container_op(
    decisiontree, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


svm_op = comp.func_to_container_op(
    svm, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


naivebayes_op = comp.func_to_container_op(
    naivebayes, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


logisticregression_op = comp.func_to_container_op(
    logisticregression, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


randomforest_op = comp.func_to_container_op(
    randomforest, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


results_op = comp.func_to_container_op(
    results, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


final_auto_snapshot_op = comp.func_to_container_op(
    final_auto_snapshot, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


@dsl.pipeline(
    name='titanic-ml-fylgn',
    description='Predict which passengers survived the Titanic shipwreck'
)
def auto_generated_pipeline(rok_workspace_aidays01_2rlcyd0k8_url='http://rok.rok.svc.cluster.local/swift/v1/50d8a2cc-a663-4a39-a455-15c0f0d50cf8/notebooks/aidays01-0_workspace-aidays01-2rlcyd0k8?version=413593cc-8852-4c61-bd92-47653e4db222'):
    pvolumes_dict = OrderedDict()

    annotations = {'rok/origin': 'http://rok.rok.svc.cluster.local/swift/v1/50d8a2cc-a663-4a39-a455-15c0f0d50cf8/notebooks/aidays01-0_workspace-aidays01-2rlcyd0k8?version=413593cc-8852-4c61-bd92-47653e4db222'}

    annotations['rok/origin'] = rok_workspace_aidays01_2rlcyd0k8_url

    vop1 = dsl.VolumeOp(
        name='create-volume-1',
        resource_name='workspace-aidays01-2rlcyd0k8',
        annotations=annotations,
        size='5Gi'
    )
    volume = vop1.volume

    pvolumes_dict['/home/jovyan'] = volume

    loaddata_task = loaddata_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    loaddata_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    loaddata_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    loaddata_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    datapreprocessing_task = datapreprocessing_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(loaddata_task)
    datapreprocessing_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    datapreprocessing_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    datapreprocessing_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    featureengineering_task = featureengineering_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(datapreprocessing_task)
    featureengineering_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    featureengineering_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    featureengineering_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)

    decisiontree_task = decisiontree_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    decisiontree_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    decisiontree_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    decisiontree_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    svm_task = svm_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    svm_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    svm_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    svm_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    naivebayes_task = naivebayes_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    naivebayes_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    naivebayes_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    naivebayes_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    logisticregression_task = logisticregression_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    logisticregression_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    logisticregression_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    logisticregression_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)

    randomforest_task = randomforest_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(featureengineering_task)
    randomforest_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    randomforest_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    randomforest_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    results_task = results_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(randomforest_task, logisticregression_task, naivebayes_task, svm_task, decisiontree_task)
    results_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    results_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    results_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    final_auto_snapshot_task = final_auto_snapshot_op(rok_workspace_aidays01_2rlcyd0k8_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(results_task)
    final_auto_snapshot_task.container.working_dir = "/home/jovyan/examples/titanic-ml-dataset"
    final_auto_snapshot_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    final_auto_snapshot_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('titanic')

    # Submit a pipeline run
    run_name = 'titanic-ml-fylgn_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
