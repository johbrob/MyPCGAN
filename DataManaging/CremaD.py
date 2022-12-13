import numpy as np

import local_vars
import pandas as pd
import numpy as np

def filterWeirdValues(dataFrame, path):
    malformed_rows = []
    for i, row in resp.iterrows():
        if not row['respLevel'].isdigit():
            # row['respLevel'] = np.NAN
            malformed_rows.append(i)
9
    dataFrame.drop(malformed_rows)
    file_path, extension =
    dataFrame.to_csv()

if __name__ == '__main__':
    # emoResp = pd.read_csv(local_vars.CREMAD_PATH + 'finishedEmoResponses.csv')
    resp = pd.read_csv(local_vars.CREMAD_PATH + 'finishedResponses.csv',
                       dtype={#'localid': str, 'pos': int, 'ans': str, 'ttr': int, 'queryType': int, 'numTries': int,
                              #'clipNum': int, 'questNum': int, 'subType': int, 'clipName': str, 'sessionNums': int,
                              'respEmo': str, 'respLevel': str, #'dispEmo': str, 'dispVal': float, 'dispLevel': str
                       })

    for i, row in resp.iterrows():
        if not row['respLevel'].isdigit():
            # print(row['respLevel'], i , row)
            row['respLevel'] = np.NAN

    for i, row in resp.iterrows():
        if not row['respLevel'].isdigit():
            print(row['respLevel'], i , row)
    # respWRWP = pd.read_csv(local_vars.CREMAD_PATH + 'finishedResponsesWithRepeatWithPractice.csv',
    #                        dtype={'respLevel': int})
    # sentence_filenames = pd.read_csv(local_vars.CREMAD_PATH + 'SentenceFilenames.csv')
    # video_demographics = pd.read_csv(local_vars.CREMAD_PATH + 'VideoDemographics.csv')
