import os
import pickle
import sklearn
import math
import pandas as pd
import geopandas as geo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.util.testing as tm
from math import sqrt
from statistics import mean
import pyproj


def class_outliers(df):
    outlier_index_list = []
    for index,row in df.iterrows():
        if index == len(df)-2:
            break
        else:
            item_1 = df.iloc[index]['class']
            item_2 = df.iloc[index+1]['class']
            item_3 = df.iloc[index+2]['class']

            if item_1 != item_2 and item_2 != item_3 and item_1 == item_3:
                outlier_index_list.append((index+1,item_2))
            else:
                pass


    for i in outlier_index_list:
        if i[1] == 0:
            df.loc[i[0],('class')] = 1
            print("Point filtered: Index position: {index} Original point: {original}, Changed to: {changed}".format(index = str(i[0]), original = str(1), changed = str(i[1])))
        else:
            df.loc[i[0],('class')] = 0
            print("Point filtered: Index position: {index} Original point: {original}, Changed to: {changed}".format(index = str(i[0]), original = str(0), changed = str(i[1])))

    return(df)




class centerline_feature_creator():
    """ Class function is meant to take the pre-processed machine learning categorized point data and
        convert to a number of list in order to generate clean geometry and alignments. Horizontal Only right now."""

    def __init__(self,
                 dataframe,
                 latitude = None,
                 longitude = None,
                 bearing_angles = None,
                 three_point_sum = None,
                 five_point_sum = None,
                 segment_length = None,
                 circonscrit_circle_radius = None,
                 oscillating_circle_radius = None,
                 output = None):

        """ Initialize some variables """
        self.dataframe = dataframe
        self.latitude = lat_long_convert(self.dataframe)[0]
        self.longitude = lat_long_convert(self.dataframe)[1]
        self.bearing_angles = bearing_function(self.dataframe)
        self.three_point_sum = three_point(self.bearing_angles)
        self.five_point_sum = five_point(self.bearing_angles)
        self.segment_length = distance_calc(self.dataframe)
        self.circonscrit_circle_radius = circon_calc(self.dataframe)
        self.oscillating_circle_radius = oscill_calc(self.dataframe)
        self.output = combine(self.dataframe['Northing'],self.dataframe['Easting'],self.latitude,self.longitude,self.dataframe['Elevation'],self.bearing_angles, self.three_point_sum,self.five_point_sum,self.segment_length,self.circonscrit_circle_radius,self.oscillating_circle_radius)


def cercle_circonscrit(T):
    (x1, y1), (x2, y2), (x3, y3) = T
    A = np.array([[x3-x1,y3-y1],[x3-x2,y3-y2]])
    Y = np.array([(x3**2 + y3**2 - x1**2 - y1**2),(x3**2+y3**2 - x2**2-y2**2)])
    if np.linalg.det(A) == 0:
        return 0
    Ainv = np.linalg.inv(A)
    X = 0.5*np.dot(Ainv,Y)
    x,y = X[0],X[1]
    r = sqrt((x-x1)**2+(y-y1)**2)
#     return (x,y),r
    return r

def bearing_finder(grouped_points):
    list_of_thetas = []
    for i in grouped_points:
            num = ((i[1][0]-i[0][0])*(i[2][0]-i[1][0])+(i[1][1]-i[0][1])*(i[2][1]-i[1][1]))
            den = math.sqrt(((i[1][0]-i[0][0])**2)+((i[1][1]-i[0][1])**2))*math.sqrt(((i[2][0]-i[1][0])**2)+((i[2][1]-i[1][1])**2))
            final_angle_calc = math.acos(num/den)*(180/math.pi)
            list_of_thetas.append(final_angle_calc)
    return(list_of_thetas)

def oscillating_circle_radius_finder(x1, y1, x2, y2, x3, y3):
        '''
        Adapted and modifed to get the unknowns for defining a parabola:
        http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
        '''
        denom = (x1-x2) * (x1-x3) * (x2-x3)
        A     = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
        B     = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
        C     = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

        p= np.poly1d([A,B,C])
        first_deriv= np.polyder(p,1)
        second_deriv= np.polyder(p,2)

        '''Return a list of:
            1. Original Polynomial
            2. First Derivative
            3. Second Derivative'''
        return [p,first_deriv,second_deriv]


'''Actual Creation of Features Follows:'''
'''takes in a pandas dataframe'''
def lat_long_convert(self):
    '''create Lat and Long points from northing easting US COORDINATES'''
    #define coord systems.
    wgs84=pyproj.CRS("EPSG:4326") # Lat Long
    NAD83=pyproj.CRS("EPSG:2263") # N E


    #lat is northing.
    geo_lat = []
    geo_long = []
    for index,row in self.iterrows():
        geo_lat.append(row['Northing'])
        geo_long.append(row['Easting'])


    converted = pyproj.transform(NAD83, wgs84, geo_lat,geo_long)

    #latitude, Longitude Tuple
    return((converted[0],converted[1]))


def bearing_function(self):
    bearing_list = [0]
    for i in range(len(self)-1):
        try:
            current_segment_bearing = ((self.iloc[i]["Easting"]),(self.iloc[i]["Northing"])),((self.iloc[i+1]["Easting"]),(self.iloc[i+1]["Northing"])),((self.iloc[i+2]["Easting"]),(self.iloc[i+2]["Northing"]))
            bearing_list.append(bearing_finder([current_segment_bearing])[0])
        except:
            #if code above wont run, put in CLOSE to zero number.
            bearing_list.append(0.01*10**-12)
    return(bearing_list)

def three_point(self):
    counter = 0
    index_list = [0,1,2]
    three_point_sum = [0]
    for i in range(len(self)):
        sum_list = [self[i],self[i+1],self[i+2]]
        three_point_sum.append(sum(sum_list))
        index_list = list(map(lambda x : x + 1, index_list))
        sum_list = []
        counter += 1
        if counter == len(self)-2:
            three_point_sum.append(0)
            break
    return(three_point_sum)

def five_point(self):
    counter = 0
    index_list = [0,1,2,3,4]
    five_point_sum = [0,0]
    for i in range(len(self)):
        sum_list = [self[i],self[i+1],self[i+2],self[i+3],self[i+4]]
        five_point_sum.append(sum(sum_list))
        index_list = list(map(lambda x : x + 1, index_list))
        sum_list = []
        counter += 1
        if counter == len(self)-5:
            five_point_sum.append(0)
            five_point_sum.append(0)
            five_point_sum.append(0)
            break
    return(five_point_sum)

def distance_calc(self):
    index_list = [0,1]
    tangent_list = []
    counter = 0
    for index,row in self.iterrows():
        xb = self.iloc[index_list[1]]['Easting']
        xa = self.iloc[index_list[0]]['Easting']
        yb = self.iloc[index_list[1]]['Northing']
        ya = self.iloc[index_list[0]]['Northing']
        tangent_length = math.sqrt(((xb-xa)**2) + ((yb-ya)**2))
        tangent_list.append(tangent_length)
        index_list = list(map(lambda x : x + 1, index_list))
        lengent_length = []
        counter += 1
        if counter == len(self.index)-1:
            tangent_list.append(mean(tangent_list))
            break
    return(tangent_list)

def circon_calc(self):
    counter = 0
    index_list = [0,1,2]
    conscrit_list = [0]
    for index,row in self.iterrows():
        T = (self.iloc[index_list[0]]['Easting'],self.iloc[index_list[0]]['Northing']),(self.iloc[index_list[1]]['Easting'],self.iloc[index_list[1]]['Northing']),(self.iloc[index_list[2]]['Easting'],self.iloc[index_list[2]]['Northing'])
        circonscrit_circle_r = cercle_circonscrit(T)
        conscrit_list.append(circonscrit_circle_r)
        circonscrit_circle_r = []
        index_list = list(map(lambda x : x + 1, index_list))
        counter += 1
        if counter == len(self.index)-2:
            conscrit_list.append(0)
            break
    return(conscrit_list)

def oscill_calc(self):
    counter = 0
    index_list = [0,1,2]
    oscillating_circle_list = [0]
    for index,row in self.iterrows():

        #define x of zero.
        x_0 = self.iloc[index_list[1]]['Easting']

        #define the working points for the calcs.
        a1,b1=[self.iloc[index_list[0]]['Easting'],self.iloc[index_list[0]]['Northing']]
        a2,b2=[self.iloc[index_list[1]]['Easting'],self.iloc[index_list[1]]['Northing']]
        a3,b3=[self.iloc[index_list[2]]['Easting'],self.iloc[index_list[2]]['Northing']]

        #this variable stores polynomial, derivative 1 and 2.
        polynomial_storage = oscillating_circle_radius_finder(a1, b1, a2, b2, a3, b3)

        first_derv_zero = polynomial_storage[1](x_0)
        second_derv_zero = polynomial_storage[2](x_0)

        #to avoid an error of div by zero make close to zero.
        if second_derv_zero == 0:
            second_derv_zero = 1.0*10**-13
        else:
            pass

        #this solves for the radius of the oscillating circle
        oscillating_circle_r = (1+((first_derv_zero)**2)**(3/2))/(abs(second_derv_zero))


        oscillating_circle_list.append(oscillating_circle_r)
        index_list = list(map(lambda x : x + 1, index_list))
        counter += 1
    #         print(index_list)
    #         print(oscillating_circle_r)
        oscillating_circle_r = []


        if counter == len(self.index)-2:
            oscillating_circle_list.append(0)
            break
    return(oscillating_circle_list)


def combine(northing,easting,latitude,longitude,elevation,bearing_angles,three_point_sum,five_point_sum,segment_length,circonscrit_circle_radius,oscillating_circle_radius):
    out_frame = None
    out_frame = pd.DataFrame(list(zip(northing,easting,latitude,longitude,elevation,bearing_angles, three_point_sum, five_point_sum,segment_length,circonscrit_circle_radius,oscillating_circle_radius)),
                  columns=['Northing','Easting','Latitude','Longitude','Elevation','bearing_angles','three_point_sum', 'five_point_sum','segment_length','circonscrit_circle_radius','oscillating_circle_radius'])
    return(out_frame)
