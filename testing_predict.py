# Data for testing model predictions.
# Headlines
x = ["Stanford Students Admit It Was Pretty Obvious Billionaire’s Dog Didn’t Get In By Itself",
     "Donald Trump Jr. Is Very Mad That Far-Right Extremists Have Lost A Platform",
     "Trump And Democrats Agree On $2 Trillion Infrastructure Deal",
     "Beto O’Rourke Blasts ‘Hatred’ Spewed By Pete Buttigieg’s Anti-Gay Hecklers",
     "JJ Abrams Announces Meryl Streep Will Take Over Role Of Chewbacca",
     "Kamala Harris Fires Back At Donald Trump For Calling Her ‘Nasty’",
     "Zoologists Thrilled After Successfully Getting Pair Of Bengal Tigers To 69 In Captivity",
     "Pete Buttigieg Is Exposing Tensions In The LGBTQ Movement",
     "K-Pop Group BTS Excited For First American Tour Since 1963 Appearance On ‘Ed Sullivan’",
     "Dozens Of Civil Rights Groups Ask Presidential Candidates To Support Letting People In Prison Vote"]

sarc_data = [a.lower() for a in x]

# Sarcasm classification
sarc_target = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

print(sarc_data)
