---
title: EDA
notebook: CS109a_Final_Project_Model2.ipynb
nav_include: 1
---

## Contents
{:.no_toc}
*  
{: toc}



```python
import pandas as pd
import numpy as np
import math
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
from time import sleep
import pandas as pd
import numpy as np
import math
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
```




```python
import pandas as pd
import numpy as np
import math
import requests
from bs4 import BeautifulSoup
from bs4 import Comment
boxscores = []
```




```python
plays = pd.read_csv('MLBPlayData.csv').drop('Unnamed: 0', axis=1)
```


to change: 
- column 9: change to two columns, with batting team's runs and opponent team's runs
- column 11: parse to three variables, curr_man_on_first, curr_man_on_second, curr_man_on_third
- column 12: parse to three variables, number of pitches in at bat, number of balls in at bat, number of strikes in at bat 
- column 13: parse to two variables for number of runs and number of outs
- column 19: parse for was_out, was_single, was_double, was_triple, was_homerun, was_strikeout, was_groundout, was_error ??, 



```python
plays.columns = ['gameID', 'date', 'stadium', 'attendance', 'inning', 'starting_pitcher', 
                 'is_starting_pitcher', 'is_away', 'batting_position', 'score', 'num_outs', 
                 'runners_on_base', 'pitch_details', 'run_out_result', 'team_at_bat', 'batter', 'pitcher', 
                 'WEC', 'wWE', 'play_description', 'team_won']
plays.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameID</th>
      <th>date</th>
      <th>stadium</th>
      <th>attendance</th>
      <th>inning</th>
      <th>starting_pitcher</th>
      <th>is_starting_pitcher</th>
      <th>is_away</th>
      <th>batting_position</th>
      <th>score</th>
      <th>...</th>
      <th>runners_on_base</th>
      <th>pitch_details</th>
      <th>run_out_result</th>
      <th>team_at_bat</th>
      <th>batter</th>
      <th>pitcher</th>
      <th>WEC</th>
      <th>wWE</th>
      <th>play_description</th>
      <th>team_won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0-0</td>
      <td>...</td>
      <td>---</td>
      <td>5,(1-2) CFBFX</td>
      <td>NaN</td>
      <td>TOR</td>
      <td>Reed Johnson</td>
      <td>Joe Saunders</td>
      <td>-6%</td>
      <td>44%</td>
      <td>Double to CF (Fly Ball to Deep CF)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0-0</td>
      <td>...</td>
      <td>-2-</td>
      <td>6,(3-2) FBFBBX</td>
      <td>O</td>
      <td>TOR</td>
      <td>Aaron Hill</td>
      <td>Joe Saunders</td>
      <td>4%</td>
      <td>48%</td>
      <td>Lineout: 3B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0-0</td>
      <td>...</td>
      <td>-2-</td>
      <td>2,(1-0) *BX</td>
      <td>RO</td>
      <td>TOR</td>
      <td>Vernon Wells</td>
      <td>Joe Saunders</td>
      <td>-4%</td>
      <td>44%</td>
      <td>Single to RF (Line Drive); Johnson Scores; out...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1-0</td>
      <td>...</td>
      <td>---</td>
      <td>2,(0-1) CX</td>
      <td>O</td>
      <td>TOR</td>
      <td>Troy Glaus</td>
      <td>Joe Saunders</td>
      <td>1%</td>
      <td>45%</td>
      <td>Lineout: 3B-1B</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0-1</td>
      <td>...</td>
      <td>---</td>
      <td>4,(1-2) BCCX</td>
      <td>NaN</td>
      <td>LAA</td>
      <td>Maicer Izturis</td>
      <td>A.J. Burnett</td>
      <td>4%</td>
      <td>49%</td>
      <td>Single to 1B (Ground Ball)</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>





```python
plays_raw = plays.copy()
```




```python
plays['runs_in_atbat'] = [0 if type(row)==float else row.count('R') for row in plays['run_out_result']]
plays['outs_in_atbat'] = [0 if type(row)==float else row.count('O') for row in plays['run_out_result']]

plays['runner_on_first']= [0 if runner[0]=='-' else 1 for runner in plays['runners_on_base']]
plays['runner_on_second']= [0 if runner[1]=='-' else 1 for runner in plays['runners_on_base']]
plays['runner_on_third']= [0 if runner[2]=='-' else 1 for runner in plays['runners_on_base']]

plays['batting_team_runs'] = [score.split('-')[0] for score in plays['score']]
plays['fielding_team_runs'] = [score.split('-')[1] for score in plays['score']]

plays["pitch_count"] = [pitch.split(',')[0] if isinstance(pitch,str) else None for pitch in plays['pitch_details']]
plays["ball_count"] = [pitch.split('(')[1].split('-')[0] if isinstance(pitch,str) else None for pitch in plays['pitch_details']]
plays["strike_count"] = [pitch.split('(')[1].split('-')[1].split(')')[0] if isinstance(pitch,str) else None for pitch in plays['pitch_details']]

plays["is_single"] = ['Single' in play for play in plays['play_description']]
plays["is_double"] = ['Double' in play for play in plays['play_description']]
plays["is_triple"] = ['Triple' in play for play in plays['play_description']]
plays["is_homerun"] = ['Homerun' in play for play in plays['play_description']]
plays["is_strikeout"] = ['Strikeout' in play for play in plays['play_description']]
plays["is_groundout"] = ['Groundout' in play for play in plays['play_description']]
plays["is_walk"] = ['Walk' in play for play in plays['play_description']]
plays["is_steal"] = [('Steal' or 'Steals' or 'steal' or 'steals') in play for play in plays['play_description']]

plays['batting_team_runs'] = plays['batting_team_runs'].apply(lambda x: int(x))
plays['fielding_team_runs'] = plays['fielding_team_runs'].apply(lambda x: int(x))
plays['run_dif']= plays['batting_team_runs'] - plays['fielding_team_runs']

plays['wWE'] = plays['wWE'].apply(lambda x: int(x.replace("%","")))

plays = plays.drop(['run_out_result','runners_on_base', 'pitch_details', 'play_description'], axis=1)

plays
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameID</th>
      <th>date</th>
      <th>stadium</th>
      <th>attendance</th>
      <th>inning</th>
      <th>starting_pitcher</th>
      <th>is_starting_pitcher</th>
      <th>is_away</th>
      <th>batting_position</th>
      <th>score</th>
      <th>...</th>
      <th>strike_count</th>
      <th>is_single</th>
      <th>is_double</th>
      <th>is_triple</th>
      <th>is_homerun</th>
      <th>is_strikeout</th>
      <th>is_groundout</th>
      <th>is_walk</th>
      <th>is_steal</th>
      <th>run_dif</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0-0</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1-0</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0-1</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>1</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0-1</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0-1</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0-1</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>0-1</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>2</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1-0</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1-0</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>3</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>4</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>4</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1-0</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>4</td>
      <td>Joe Saunders</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1-0</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>4</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>4</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0-1</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>Friday, September 8, 2006</td>
      <td>Angel Stadium of Anaheim</td>
      <td>42,259</td>
      <td>4</td>
      <td>A.J. Burnett</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1-1</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>281882</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>5</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2-1</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281883</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>5</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2-1</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281884</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>5</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281885</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>5</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2-1</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281886</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>5</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281887</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1-4</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281888</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1-4</td>
      <td>...</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281889</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281890</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281891</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>4-1</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281892</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281893</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>6</td>
      <td>Luis Severino</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281894</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>7</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281895</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>7</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281896</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>7</td>
      <td>Chris Tillman</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281897</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>7</td>
      <td>Luis Severino</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4-1</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281898</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>7</td>
      <td>Luis Severino</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281899</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>7</td>
      <td>Luis Severino</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281900</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281901</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1-4</td>
      <td>...</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281902</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>1-4</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281903</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281904</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Luis Severino</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281905</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Luis Severino</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>4-1</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281906</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>8</td>
      <td>Luis Severino</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>4-1</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>281907</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>9</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1-4</td>
      <td>...</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281908</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>9</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>1-4</td>
      <td>...</td>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281909</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>9</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>1-4</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281910</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>9</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>1-4</td>
      <td>...</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
    <tr>
      <th>281911</th>
      <td>3537</td>
      <td>Tuesday, May 3, 2016</td>
      <td>Oriole Park at Camden Yards</td>
      <td>16,083</td>
      <td>9</td>
      <td>Chris Tillman</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1-4</td>
      <td>...</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-3</td>
    </tr>
  </tbody>
</table>
<p>281912 rows × 36 columns</p>
</div>





```python
import matplotlib
import matplotlib.pyplot as plt

winning_df = plays.loc[plays['team_won']==1]
losing_df = plays.loc[plays['team_won']==0]

winning_nonstarter = winning_df.loc[winning_df['is_starting_pitcher']==0]
losing_nonstarter = losing_df.loc[losing_df['is_starting_pitcher']==0]

winning_nonstarter = winning_nonstarter.drop_duplicates(subset='gameID', keep='first')
losing_nonstarter = losing_nonstarter.drop_duplicates(subset='gameID', keep='first')

winning_avg_nonstarter_inning = np.mean(winning_nonstarter['inning'])
losing_avg_nonstarter_inning = np.mean(losing_nonstarter['inning'])

fig, ax = plt.subplots(figsize=(9,6))
ax.bar([1,2], [winning_avg_nonstarter_inning, losing_avg_nonstarter_inning], tick_label=['Winning', 'Losing'])
ax.set_ylabel('Inning')
ax.set_xlabel('Winning or Losing Team')
ax.set_title('Average Inning Starting Pitcher Leaves: Winning vs. Losing Team')
plt.show()
```



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_7_0.png)




```python
same_score_df = plays.loc[plays['run_dif']==0]

bases_empty = same_score_df.loc[(same_score_df['runner_on_first']==0) &
                               (same_score_df['runner_on_second']==0) & (same_score_df['runner_on_third']==0)]
just_first = same_score_df.loc[(same_score_df['runner_on_first']==1) & 
                               (same_score_df['runner_on_second']==0) & (same_score_df['runner_on_third']==0)]
just_second = same_score_df.loc[(same_score_df['runner_on_first']==0) & 
                               (same_score_df['runner_on_second']==1) & (same_score_df['runner_on_third']==0)]
just_third = same_score_df.loc[(same_score_df['runner_on_first']==0) & 
                               (same_score_df['runner_on_second']==0) & (same_score_df['runner_on_third']==1)]
first_second = same_score_df.loc[(same_score_df['runner_on_first']==1) & 
                               (same_score_df['runner_on_second']==1) & (same_score_df['runner_on_third']==0)]
second_third = same_score_df.loc[(same_score_df['runner_on_first']==0) & 
                               (same_score_df['runner_on_second']==1) & (same_score_df['runner_on_third']==1)]
first_third = same_score_df.loc[(same_score_df['runner_on_first']==1) & 
                               (same_score_df['runner_on_second']==0) & (same_score_df['runner_on_third']==1)]
bases_loaded = same_score_df.loc[(same_score_df['runner_on_first']==1) & 
                               (same_score_df['runner_on_second']==1) & (same_score_df['runner_on_third']==1)]

runners_win_prob = {}
runners_win_prob['Bases_Empty']= np.mean(bases_empty['wWE'])
runners_win_prob['Just_First']= np.mean(just_first['wWE'])
runners_win_prob['Just_Second']= np.mean(just_second['wWE'])
runners_win_prob['Just_Third']= np.mean(just_third['wWE'])
runners_win_prob['First_and_Second']= np.mean(first_second['wWE'])
runners_win_prob['First_and_Third']= np.mean(first_third['wWE'])
runners_win_prob['Second_and_Third']= np.mean(second_third['wWE'])
runners_win_prob['Bases_Loaded']= np.mean(bases_loaded['wWE'])

result_list = sorted([[k, v] for k, v in runners_win_prob.items()], key = lambda x: x[0])

runners = [x[0] for x in result_list]
win_prob = [x[1] for x in result_list]

fig, ax = plt.subplots(figsize=(15,8))
ax.bar(range(8), win_prob, tick_label=runners)
ax.set_ylabel('Win Probability of Eventual Winning Team')
ax.set_xlabel('Runner Configuration')
ax.set_title('Average Win Probability vs. Runner on Base Configuration (when score is equal)')
plt.show()
```


    [50.927874907658214, 59.789439374185136, 54.302051594556168, 56.044411908247923, 51.957037874505367, 52.872322899505768, 52.587276550998951, 56.274834437086092]



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_8_1.png)




```python
import seaborn as sns

win_prob_by_score = pd.pivot_table(plays, values='wWE', index=['fielding_team_runs'], columns=['batting_team_runs'])
win_prob_by_score
fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(win_prob_by_score)
ax.invert_yaxis()
ax.set_title('Win Probability of Eventual Winning Team vs. Score')
plt.show()
```



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_9_0.png)




```python
plays['home_team_won'] = np.where(plays['team_won'] == plays['is_away'], 0, 1)

plays_first_play = plays.drop_duplicates(subset='gameID', keep='first')
plays_last_play = plays.drop_duplicates(subset='gameID', keep='last')

by_home_away = plays_last_play.groupby('home_team_won').size()

fig, ax = plt.subplots(figsize=(10,6))
ax.bar([0,1], by_home_away)
ax.set_xticks([0,1])
ax.set_xticklabels(['Away', 'Home'])
ax.set_title('Games Won by Home/Away Teams')
ax.set_ylabel('Number of Games')
plt.show()
```



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_10_0.png)




```python
rows=[]
for i in range(1, len(plays['gameID'])):
    if (plays['gameID'][i] != plays['gameID'][i-1]):
        Runsone=plays['batting_team_runs'][i-1]+plays['runs_in_atbat'][i-1]
        Runstwo=plays['fielding_team_runs'][i-1]
        Win=plays['team_won'][i-1]
        FieldWin=plays['team_won'][i-1]*-1+1
        row=[plays['gameID'][i-1], Runsone, Runstwo, Win, FieldWin]
        rows.append(row)
        
columns=['GameID', 'Batting Runs','Fielding Runs', 'Win', 'FieldWin']
rundata=pd.DataFrame(rows, columns=columns)
```




```python
frames=[rundata['Batting Runs'], rundata['Fielding Runs']]
frames2=[rundata['Win'], rundata['FieldWin']]
stacked=pd.concat(frames)
stackedwin=pd.concat(frames2)

final = pd.concat([stacked,stackedwin], axis=1)
final.columns=['Runs', 'Win']

build=[]
matchedruns=[]
for runs in range(22):
    k=final[final['Runs']==runs]
    pct=np.sum(k['Win'])/len(k)
    build.append(pct)
    matchedruns.append(runs)
```




```python
plt.scatter(matchedruns, build)
extraticks=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
plt.xlabel('Runs')
plt.ylabel('Probability of Winning')
plt.title('Probability of Winning Given Scoring runs')
plt.xticks(extraticks)
plt.show()
```



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_13_0.png)




```python
newdata= pd.concat([plays["run_dif"], plays["batting_position"], plays["team_won"]], axis=1)
newdata

newrows=[]
for j in range(1,10):
    for i in range(-10,11):
        z=newdata[newdata['run_dif']==i]
        Z=z[z['batting_position']==j]
        percent=np.sum(Z['team_won'])/len(Z)
        newrow=(j, i, percent)
        newrows.append(newrow)
columns1=['Batting Pos', 'Run Differential', 'Win Probability']
finall=pd.DataFrame(newrows, columns=columns1)

matrix = finall.pivot( 'Run Differential','Batting Pos', 'Win Probability')
matrix
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Batting Pos</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
    <tr>
      <th>Run Differential</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-10</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>-9</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>-8</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>-7</th>
      <td>0.005780</td>
      <td>0.002933</td>
      <td>0.005797</td>
      <td>0.011561</td>
      <td>0.011940</td>
      <td>0.011628</td>
      <td>0.008571</td>
      <td>0.005970</td>
      <td>0.002959</td>
    </tr>
    <tr>
      <th>-6</th>
      <td>0.023091</td>
      <td>0.024209</td>
      <td>0.017375</td>
      <td>0.019380</td>
      <td>0.018975</td>
      <td>0.023622</td>
      <td>0.022133</td>
      <td>0.030303</td>
      <td>0.026465</td>
    </tr>
    <tr>
      <th>-5</th>
      <td>0.049180</td>
      <td>0.046674</td>
      <td>0.045506</td>
      <td>0.048919</td>
      <td>0.048220</td>
      <td>0.045238</td>
      <td>0.045667</td>
      <td>0.042118</td>
      <td>0.041162</td>
    </tr>
    <tr>
      <th>-4</th>
      <td>0.092593</td>
      <td>0.093473</td>
      <td>0.092639</td>
      <td>0.090909</td>
      <td>0.085622</td>
      <td>0.083736</td>
      <td>0.078624</td>
      <td>0.075868</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>-3</th>
      <td>0.141863</td>
      <td>0.140294</td>
      <td>0.142540</td>
      <td>0.137127</td>
      <td>0.127919</td>
      <td>0.124190</td>
      <td>0.131766</td>
      <td>0.133483</td>
      <td>0.132488</td>
    </tr>
    <tr>
      <th>-2</th>
      <td>0.230312</td>
      <td>0.231840</td>
      <td>0.229735</td>
      <td>0.231949</td>
      <td>0.222180</td>
      <td>0.225181</td>
      <td>0.220286</td>
      <td>0.212650</td>
      <td>0.210965</td>
    </tr>
    <tr>
      <th>-1</th>
      <td>0.375095</td>
      <td>0.378107</td>
      <td>0.373858</td>
      <td>0.373384</td>
      <td>0.364123</td>
      <td>0.354290</td>
      <td>0.349150</td>
      <td>0.351070</td>
      <td>0.351734</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.515577</td>
      <td>0.517555</td>
      <td>0.515354</td>
      <td>0.520568</td>
      <td>0.527435</td>
      <td>0.524248</td>
      <td>0.520953</td>
      <td>0.519365</td>
      <td>0.523621</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.691130</td>
      <td>0.682659</td>
      <td>0.683290</td>
      <td>0.682032</td>
      <td>0.672054</td>
      <td>0.664665</td>
      <td>0.671806</td>
      <td>0.682590</td>
      <td>0.681231</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.790724</td>
      <td>0.795885</td>
      <td>0.807600</td>
      <td>0.802185</td>
      <td>0.785269</td>
      <td>0.787173</td>
      <td>0.785992</td>
      <td>0.790698</td>
      <td>0.790847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.883501</td>
      <td>0.886552</td>
      <td>0.881087</td>
      <td>0.872866</td>
      <td>0.881708</td>
      <td>0.874258</td>
      <td>0.873551</td>
      <td>0.872050</td>
      <td>0.878279</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.912338</td>
      <td>0.912324</td>
      <td>0.913469</td>
      <td>0.912541</td>
      <td>0.920260</td>
      <td>0.915296</td>
      <td>0.919936</td>
      <td>0.915200</td>
      <td>0.907916</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.932530</td>
      <td>0.943210</td>
      <td>0.939589</td>
      <td>0.950980</td>
      <td>0.948941</td>
      <td>0.959259</td>
      <td>0.947837</td>
      <td>0.947619</td>
      <td>0.936980</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.982036</td>
      <td>0.977413</td>
      <td>0.970356</td>
      <td>0.964981</td>
      <td>0.967557</td>
      <td>0.965779</td>
      <td>0.969697</td>
      <td>0.967495</td>
      <td>0.974257</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.978788</td>
      <td>0.968750</td>
      <td>0.980769</td>
      <td>0.983498</td>
      <td>0.977124</td>
      <td>0.977199</td>
      <td>0.975155</td>
      <td>0.981481</td>
      <td>0.979228</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.981735</td>
      <td>0.986486</td>
      <td>0.990566</td>
      <td>0.985149</td>
      <td>0.995392</td>
      <td>0.986726</td>
      <td>0.982684</td>
      <td>0.982833</td>
      <td>0.986547</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.984252</td>
      <td>0.983740</td>
      <td>0.966942</td>
      <td>0.973913</td>
      <td>0.967213</td>
      <td>0.985185</td>
      <td>0.992126</td>
      <td>0.983471</td>
      <td>0.984375</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.986667</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.987500</td>
      <td>0.986486</td>
      <td>0.986301</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>





```python
import seaborn as sns
fig = plt.figure(figsize=(21,9))
r = sns.heatmap(matrix)
r.set_title("Run Differential vs Batting Position")
plt.show()
```


    /opt/anaconda3/lib/python3.6/site-packages/matplotlib/font_manager.py:1297: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_15_1.png)




```python
newdata= pd.concat([plays["run_dif"], plays["inning"], plays["team_won"]], axis=1)
newdata

newrows=[]
for j in range(1,10):
    for i in range(-10,11):
        z=newdata[newdata['run_dif']==i]
        Z=z[z['inning']==j]
        if len(Z)==0:
            continue
        
        percent=np.sum(Z['team_won'])/len(Z)
        newrow=(j, i, percent)
        newrows.append(newrow)
columns1=['Inning', 'Run Differential', 'Win Probability']
finall=pd.DataFrame(newrows, columns=columns1)

matrix = finall.pivot( 'Run Differential','Inning', 'Win Probability')
matrix

fig = plt.figure(figsize=(21,9))
r = sns.heatmap(matrix,cmap="Blues")
r.set_title("Run Differential vs Inning")
plt.show()
```


    /opt/anaconda3/lib/python3.6/site-packages/matplotlib/font_manager.py:1297: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))



![png](CS109a_Final_Project_EDA_files/CS109a_Final_Project_EDA_16_1.png)

