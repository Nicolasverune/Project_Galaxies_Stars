import pandas as pd
import warnings

galaxies = pd.read_csv("Skyserver_GALAXY.csv", skiprows = 1)
stars = pd.read_csv("Skyserver_STAR.csv", skiprows = 1)
#gal_star = pd.read_csv("ugriz_gal_star.csv", skiprows = 1)

warnings.simplefilter(action='ignore', category=FutureWarning)

def column_remover(DataPanda) :
    
    #a = set()
    list_of_del = []
    for column in DataPanda:
        
        #a.add((column,len(DataPanda[column].unique())))            
        
        if column == 'type' : 
            break 
        if len(DataPanda[column].unique()) < 990 :            
            if -1000 in DataPanda[column].unique() or -9999 in DataPanda[column].unique() or "NaN" in DataPanda[column].unique() or None in DataPanda[column].unique() :
                new_col = DataPanda[column][DataPanda[column] != -9999 ]
                new_col = new_col[new_col != -1000]
                new_col = new_col[new_col != "NaN"]
                new_col = new_col[new_col != None]
                if len(new_col.unique()) < len(new_col)-10 : 
                    del DataPanda[column]
                    list_of_del.append(column)
                else :
                    del DataPanda[column]
                    list_of_del.append(column)

        elif "ID" in column or "_id_" in column.lower() or column[-2:] == "ID" or column[-3:].lower() == "_id" : 
            del DataPanda[column]
            list_of_del.append(column) 

        elif "name" in column.lower() or "image" in column.lower() or "img" in column.lower() : 
            del DataPanda[column]
            list_of_del.append(column)

        elif "err" in column.lower() or "error" in column.lower():
            del DataPanda[column]
            list_of_del.append(column)    

    return list_of_del

def row_remover(DataPanda) : 
        
    a = set()
    row = 0
    for index,rows in DataPanda.iterrows() : 
        count = 0
        for cell in rows  : 
            if cell == -1000 or cell == -9999 or cell == "NaN" or cell is None: 
                count += 1        
        if count > 5 :
            DataPanda = DataPanda.drop([row])            
        a.add(count)
        row += 1
        
    return DataPanda

def column_remover_with_list(DataPanda, RList) : 
    for column in RList : 
        if column in DataPanda : 
            del DataPanda[column]

"""
print(gal_star.shape)
gal_star = row_remover(gal_star)
print(gal_star.shape)
gal_star.to_csv("stat_gal_star_clean.csv")
"""


print(galaxies.shape)
L1 = column_remover(galaxies)
print(galaxies.shape)
galaxies = row_remover(galaxies)
print(galaxies.shape)

print("------------")

print(stars.shape)
L2 = column_remover(stars)
print(stars.shape)
stars = row_remover(stars)
print(stars.shape)

print("------")
column_remover_with_list(galaxies,L2)
column_remover_with_list(stars,L1)

print("---------")
print(galaxies.shape)
print(stars.shape)
#still some mistakes delteing to much columns
#the "1000" is a bit high
galaxies.to_csv("galaxies_clean.csv")
stars.to_csv("stars_clean.csv")

if "u" in galaxies and "g" in galaxies and "i" in galaxies and "r" in galaxies and "z" in galaxies  : 
    print("yes") 
if "u" in stars and "g" in stars and "i" in stars and "r" in stars and "z" in stars  : 
    print("yes")





































