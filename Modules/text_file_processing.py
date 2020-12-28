import Modules.centerline_feature_creator
import Modules.geometry_cleaner
import pickle
import pandas as pd

def text_file_processing(survey_file):
    '''step 1'''
    #import the survey text file and turn into a pandas dataframe
    df = pd.read_csv(survey_file, sep=",", header='infer')
    df.columns = ["Northing", "Easting", "Elevation"]
    df['Elevation'] = 0

    '''step 2'''
    #calculate all geometric relationships for raw points which match model features.
    r = Modules.centerline_feature_creator.centerline_feature_creator(df)
    df = r.output

    '''step 3'''
    #load the model for point categorization and run on processing dataframe.
    loaded_model = pickle.load(open('/Users/connortluck/Desktop/Roadway_Geometry_Designed_Fit/Modules/finalized_model.sav', 'rb'))

    #create the fatures which match the premade model.
    x = df[["bearing_angles",'three_point_sum','five_point_sum','segment_length','circonscrit_circle_radius','oscillating_circle_radius']]

    #predict the class
    y_pred = loaded_model.predict(x)

    #append the class to the dataframe
    df['class'] = y_pred

    #filter outliers
    df = Modules.centerline_feature_creator.class_outliers(df)

    '''step 4'''
    #Output processed geometry, and cleaned data points.
    sc = Modules.geometry_cleaner.Processed_Geometry(df)
    cleaned = Modules.geometry_cleaner.cleaned(sc)

    '''step 5'''
    #output dictionaries for front end use.
    cleaned_output = {'clean x coordinates': cleaned.cleaned_x,
                 'clean y coordinates': cleaned.cleaned_y,
                 'output_all': cleaned.output,
                 'cleaned tangent': cleaned.tangent_dataframe}

    output_geometry_data = {'Original DataFrame of Raw Points': sc.dataframe,
                        'list of curves': sc.list_of_curves,
                        'list of tangents': sc.list_of_tangents,
                        'best fit radius list': sc.best_fit_radius_list,
                        'best fit tangent list': sc.best_fit_tangent_list,
                        'points of intersection list': sc.Points_of_intersection_list,
                        'curve estiamted centers': sc.curve_estimated_centers}

    # #save locally
    # a_file = open("data.pkl", "wb")
    # b_file = open("data.pkl", "wb")
    #
    # pickle.dump(cleaned_output, a_file)
    # a_file.close()
    #
    # pickle.dump(output_geometry_data, b_file)
    # b_file.close()
    #
    # a_file = open("data.pkl", "rb")
    # output = pickle.load(a_file)
    # print(output)
    #
    # b_file = open("data.pkl", "rb")
    # output = pickle.load(a_file)
    # print(output)



    print(cleaned_output.keys(),output_geometry_data.keys())
    return(sc,cleaned,cleaned_output,output_geometry_data)
