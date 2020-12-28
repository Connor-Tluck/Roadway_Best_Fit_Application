import Modules
import geopandas as geo
import pandas as pd
import sklearn
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.util.testing as tm
from statistics import mean
import numpy as np
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pprint
import shapely as sh
from scipy import optimize
from math import sqrt
import shapely as sh
import json

from Modules.centerline_feature_creator import lat_long_convert

class Processed_Geometry():
    """ Class function is meant to take the pre-processed machine learning categorized point data and
        convert to a number of list in order to generate clean geometry and alignments. Horizontal Only right now."""

    def __init__(self,
                 dataframe,
                 list_of_curves = None,
                 list_of_tangents = None,
                 best_fit_radius_list = None,
                 best_fit_tangent_list = None,
                 Points_of_intersection_list = None,
                 curve_estimated_centers = None,
                 curve_geometry_all = None):

        """ Initialize some variables """
        self.dataframe = dataframe
        self.list_of_tangents = Processed_Geometry.tangent_sort(self.dataframe)[0]
        self.list_of_curves = Processed_Geometry.tangent_sort(self.dataframe)[1]
        self.curve_estimated_centers = Processed_Geometry.curve_center_finder(self.list_of_curves)
        self.best_fit_radius_list = Processed_Geometry.radius_fit(self.list_of_curves,self.curve_estimated_centers)
        self.best_fit_tangent_list = Processed_Geometry.best_fit_tangents(self.list_of_tangents)
        self.curve_geometry_all = Processed_Geometry.curve_geometry_combined(radius = self.best_fit_radius_list,estimated_center = self.curve_estimated_centers)
        self.Points_of_intersection_list = Processed_Geometry.list_of_PI(list_of_curves = self.list_of_curves, curve_geometry_all = self.curve_geometry_all)



    def tangent_sort(self):
        '''this function splits datapoints into curves and tangent groups'''
        '''need to revisit in order to change all code to filter out and import curve and tangent data as dataframes, currently just returns
            single points of lat and long'''
        lt=[]
        lc=[]
        lt_f = []
        lc_f = []

        for i,row in self.iterrows():
            if i == len(self)-1:
                '''this part is key because it appends the final workings lists when there are no other comparisons coming up.'''
                if len(lt_f[-1::]) != 0:
                    lt_f.append(lt)
                    return(lt_f,lc_f)
                elif len(lc_f[-1::]) != 0:
                    lc_f.append(lc)
                    return(lt_f,lc_f)

            else:
                if self.iloc[i]['class'] == 0:
                    lt.append(tuple(self.iloc[i][['Easting','Northing']]))

                    if self.iloc[i+1]['class'] == 0:
                        pass
                    else:
                        lt_f.append(lt)
                        lt = []
                else:
                    lc.append(tuple(self.iloc[i][['Easting','Northing']]))

                    if self.iloc[i+1]['class'] == 1:
                        pass
                    else:
                        lc_f.append(lc)
                        lc = []


    def best_fit_tangents(self):
        '''add this later, using PI and curve info not by calcing independ'''
        pass
# #         '''Fits a polynomial to the points found on each tangent.'''
# #         best_fit_tangent_list = []
# #         xs=[]
# #         ys=[]
# #         for i in self:
#
#
# #             xs.append(i[0][0])
# #             xs.append(i[-1::][0][0])
# #             ys.append(i[0][1])
# #             ys.append(i[-1::][0][1])
#
# #             tan_coefficients = tuple(Processed_Geometry.best_fit_slope_and_intercept(xs,ys))
# #             best_fit_tangent_list.append(tan_coefficients)
# #             xs = []
# #             ys = []
# #         return(best_fit_tangent_list)

    def curve_geometry_combined(radius,estimated_center):
        #self is best fit radius list
        #then append from curve_estimated_centers
        working_append = []
        curve_geometries = []
        for i in range(len(radius)):
            #append_estimated_radius, unfiltered points
            working_append.append(radius[i])

            #append estimated center
            working_append.append(estimated_center[i])


            #append to main list and empty working list
            curve_geometries.append(working_append)
            working_append = []

        return(curve_geometries)

    def list_of_PI(list_of_curves,curve_geometry_all):
        #cycle through the list of curves
        '''Creates a list of PI Points by finding points of intersection of each line'''
        '''stores tangent equations as well, PI, Tangent IN tangent OUT'''
        list_of_PI = []
        for i in range(len(list_of_curves)):

            #calc the slope between first point and center
            first_curve_point = list_of_curves[i][0]
            center_point = curve_geometry_all[i][1]
            radius = curve_geometry_all[i][0]

            y2 = center_point[1]
            x2 = center_point[0]
            y1 = first_curve_point[1]
            x1 = first_curve_point[0]

            '''slope and b for first point and center'''
            m1 = ((y2-y1)/(x2-x1))
            #opposite reciprical
            perp_m1 = -m1**(-1)
            b1 = y1 - (perp_m1*x1)


            #calc the slope between last point and center
            last_curve_point = list_of_curves[i][-1]
            center_point = curve_geometry_all[i][1]
            radius = curve_geometry_all[i][0]


            y2 = center_point[1]
            x2 = center_point[0]
            y1 = last_curve_point[1]
            x1 = last_curve_point[0]


            '''slope and b for last point and center'''
            m2 = ((y2-y1)/(x2-x1))
            #opposite reciprical
            perp_m2 = -m2**(-1)

            b2 = y1 - (perp_m2*x1)

            '''point of intersection calculation'''

            x = ((b2-b1)/(perp_m1-perp_m2))
            y = perp_m1*(x) + b1

            list_of_PI.append(((x,y),(perp_m1,b1),(perp_m2,b2)))

        return(list_of_PI)


    def calc_R(x,y, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f(c, x, y):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = Processed_Geometry.calc_R(x, y, *c)
        return Ri - Ri.mean()

    def leastsq_circle(x,y,curve_estimated_centers):
        # coordinates of the barycenter

        x_m = np.mean(x)
        y_m = np.mean(y)
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(Processed_Geometry.f, center_estimate, args=(x,y))
        xc, yc = center
        Ri       = Processed_Geometry.calc_R(x, y, *center)
        R        = Ri.mean()
        residu   = np.sum((Ri - R)**2)
        return xc, yc, R, residu


    def curve_center_finder(self):

        from statistics import mean
        working_center_estimate = []
        parsed_center_estimate = []
        final_center_estimate = []

        '''scan through list of curves'''
        for curve in self:

            '''find midpoint of each line'''
            for point in range(len(curve)-2):
                xa = curve[point][0]
                xb = curve[point+1][0]
                xc = curve[point+2][0]
                ya = curve[point][1]
                yb = curve[point+1][1]
                yc = curve[point+2][1]

                mid_AB = ((xa+xb)/2),((ya+yb)/2)
                mid_BC = ((xb+xc)/2),((yb+yc)/2)
                slope_AB = (yb-ya)/(xb-xa)
                slope_BC = (yc-yb)/(xc-xb)
                slope_perp_AB = -(slope_AB)**(-1)
                slope_perp_BC = -(slope_BC)**(-1)
                b_AB = ((-slope_perp_AB)*(mid_AB[0]))+(mid_AB[1])
                b_BC = ((-slope_perp_BC)*(mid_BC[0]))+(mid_BC[1])


                '''solve for x and y intersection and take mean'''

                x = (b_BC-b_AB)/(slope_perp_AB-slope_perp_BC)
                y = slope_perp_AB*(x) + b_AB

                working_center_estimate.append((x,y))
            parsed_center_estimate.append(working_center_estimate)
            working_center_estimate = []

        for i in parsed_center_estimate:
            working_df = pd.DataFrame(i, columns = ["Northing","Easting"])

            #process estimates to remove outliers.
            Q1 = working_df.quantile(0.25)
            Q3 = working_df.quantile(0.75)
            IQR = Q3 - Q1
            processed_df = working_df[~((working_df < (Q1 - 1.0 * IQR)) |(working_df > (Q3 + 1.0 * IQR))).any(axis=1)]

            #take the mean of all the filtered points which has removed outliers.
            final_center_estimate.append((mean(processed_df["Northing"]),mean(processed_df["Easting"])))
        return(final_center_estimate)

    def radius_fit (list_of_curves,curve_estimated_centers):
        list_of_radii = []
        working_radius_list = []
        for curve in range(len(list_of_curves)):
            for point in list_of_curves[curve]:

                circle_center_x= curve_estimated_centers[curve][0]
                circle_center_y=curve_estimated_centers[curve][1]
                x0 = point[0]
                y0 = point[1]


                estimated_radius = sqrt((circle_center_x - x0)**2 + (circle_center_y - y0)**2)
                working_radius_list.append(estimated_radius)
            list_of_radii.append(mean(working_radius_list))
            working_radius_list=[]
        return(list_of_radii)

class cleaned():
    """ Class function is meant to take the pre-processed machine learning categorized point data and
        convert to a number of list in order to generate clean geometry and alignments. Horizontal Only right now."""

    def __init__(self,
                 dataframe,
                 cleaned_x = None,
                 cleaned_y = None,
                 cc_df = None,
                 latitude = None,
                 longitude = None,
                 tan_best_fit = None,
                 tangent_clean_pts = None,
                 tangent_dataframe = None,
                 output = None,
                 key_points = None):

        """ Initialize some variables """
        self.dataframe = dataframe
        self.cleaned_x = cleaned.clean_curve_x(dataframe.list_of_curves)
        self.cleaned_y = cleaned.clean_curve_y(self.cleaned_x,self.dataframe.list_of_curves,self.dataframe.best_fit_radius_list,self.dataframe.curve_geometry_all)
        self.cc_df = cleaned.cc_df_create(self.cleaned_y)
        self.latitude = lat_long_convert(self.cc_df)[0]
        self.longitude = lat_long_convert(self.cc_df)[1]
        self.tan_best_fit = cleaned.tangent_equations(self.dataframe.Points_of_intersection_list)
        self.tangent_clean_pts = cleaned.clean_tangent_x(self.cleaned_y,self.tan_best_fit)
        self.tangent_dataframe = cleaned.clean_tangent_df(self.tangent_clean_pts)
        self.output = cleaned.combine(list(self.cc_df['Northing']),list(self.cc_df['Easting']),self.latitude,self.longitude)
        self.key_points = cleaned.key_points(self.dataframe.Points_of_intersection_list,self.dataframe.curve_estimated_centers)

    def clean_curve_x(dirty_curves):
        clean_x_pts_curve=[]
        #use linspace to generate clean x points with first and last cooridates.
        for curves in dirty_curves:
            clean_x_curve = np.linspace(curves[0][0],curves[-1][0],30)
            clean_x_pts_curve.append(clean_x_curve)
        return(clean_x_pts_curve)


    def clean_curve_y(clean_x_pts_curve,list_of_curves,radii,geometry_all):
        clean_y_pts_curve = []
        append_y_pts_curve = []

        #create a list of shapley linear strings to distance to calcs
        shapley_curves=[]
        for c in list_of_curves:
            shape_list = sh.geometry.LineString(c)
            shapley_curves.append(shape_list)


        for curve in range(len(clean_x_pts_curve)):

            rough_curve_object_comparison = shapley_curves[curve]

            r = radii[curve]
            xc = geometry_all[curve][1][0]
            yc = geometry_all[curve][1][1]

            for point in clean_x_pts_curve[curve]:

                y1 = yc + sqrt((r**2)-((point-xc)**2))
                y2 = yc - sqrt((r**2)-((point-xc)**2))

                point_1 = sh.geometry.point.Point(point,y1)
                point_2 = sh.geometry.point.Point(point,y2)

                clean_point_min_distance_1 = point_1.distance(rough_curve_object_comparison)
                clean_point_min_distance_2 = point_2.distance(rough_curve_object_comparison)

                if clean_point_min_distance_1 < clean_point_min_distance_2:
                    append_y_pts_curve.append((point,y1))
                else:
                    append_y_pts_curve.append((point,y2))

            clean_y_pts_curve.append(append_y_pts_curve)
            append_y_pts_curve = []
        return(clean_y_pts_curve)


    def cc_df_create(cleaned_y):
        cleaned_curves = []
        for i in cleaned_y:
            for g in i:
                cleaned_curves.append(g)

        cc_df = pd.DataFrame(data = cleaned_curves,columns = ["Easting","Northing"])
        return(cc_df)

    def combine(northing,easting,latitude,longitude):
        out_frame = None
        out_frame = pd.DataFrame(list(zip(northing,easting,latitude,longitude)),
                      columns=['Northing','Easting','Latitude','Longitude'])
        return(out_frame)

    def tangent_equations(pi_list):
        best_fit_tangent_equations = []
        for control_point in range(len(pi_list)-1):
            y2 = pi_list[control_point+1][0][1]
            x2 = pi_list[control_point+1][0][0]
            y1 = pi_list[control_point][0][1]
            x1 = pi_list[control_point][0][0]

            m = ((y2-y1)/(x2-x1)) #solve for slope
            b = y1 - (m*x1) #solve for b

            best_fit_tangent_equations.append((m,b)) #slope,b tuple
        return(best_fit_tangent_equations)


    def clean_tangent_x(clean_curve_points,best_fit_tangent_equations):
        tangent_clean_x_pts = []
        working_pts = []
        tangent_clean_pts = []
        #starting point and stop point == first and last points on each curve.
        for curve in range(len(clean_curve_points)-1):
            #last point of curve 1
            x1 = clean_curve_points[curve][-1][0]
            y1 = clean_curve_points[curve][-1][1]

            #first point of curve 2
            x2 = clean_curve_points[curve+1][0][0]
            y2 = clean_curve_points[curve+1][0][1]

            clean_x = np.linspace(x1,x2,50)
            for point in clean_x:
            #y = mx+b
                m = best_fit_tangent_equations[curve][0]
                b = best_fit_tangent_equations[curve][1]
                clean_y = (m*point) + b
                working_pts.append((point,clean_y))
            tangent_clean_pts.append(working_pts)
            working_pts=[]
        return(tangent_clean_pts)

    def clean_tangent_df(tangent_clean_pts):
        flat_list = []
        for sublist in tangent_clean_pts:
            for item in sublist:
                flat_list.append(item)

        easting_values = [x[0] for x in flat_list]
        northing_values = [y[1] for y in flat_list]

        df1 = pd.DataFrame()
        df1['Easting'] = easting_values
        df1['Northing'] = northing_values

        '''process lat and long'''
        lat_long_df1 = lat_long_convert(df1)
        df1_lat = lat_long_df1[0]
        df1_long = lat_long_df1[1]

        df1['Latitude'] = df1_lat
        df1['Longitude'] = df1_long

        return(df1)


    def key_points(points_of_intersection,estimated_center):
        from Modules.centerline_feature_creator import lat_long_convert

        # circle_center_points_x = [x[0] for x in sc.best_fit_radius_list]
        pi_x = [x[0][0] for x in points_of_intersection]
        pi_y = [y[0][1] for y in points_of_intersection]
        point_type_1 = ['point_of_intersection' for x in pi_y]
        estimated_center_x = [x[0] for x in estimated_center]
        estimated_center_y = [y[1] for y in estimated_center]
        point_type_2 = ['estimated_curve_center' for x in estimated_center_y]

        df1 = pd.DataFrame()
        df1['Easting'] = pi_x
        df1['Northing'] = pi_y
        df1['Point Type'] = point_type_1

        '''process lat and long'''
        lat_long_df1 = lat_long_convert(df1)
        df1_lat = lat_long_df1[0]
        df1_long = lat_long_df1[1]

        df1['Latitude'] = df1_lat
        df1['Longitude'] = df1_long

        df2 = pd.DataFrame()
        df2['Easting'] = estimated_center_x
        df2['Northing'] = estimated_center_y
        df2['Point Type'] = point_type_2

        '''process lat and long'''
        lat_long_df2 = lat_long_convert(df2)
        df2_lat = lat_long_df2[0]
        df2_long = lat_long_df2[1]

        df2['Latitude'] = df2_lat
        df2['Longitude'] = df2_long

        result = df1.append(df2)
        return(result)
