def convert_feature_to_text(column_feature):
    text = f"""
            column={column_feature['column_name']} 
            type={column_feature['type']} 
            missing={round(column_feature['missing_percentage'],2)} 
            unique={column_feature['unique_values']}
            """
    
    if column_feature["type"] == "numeric":
        text += f""" 
        mean={column_feature.get('mean')} 
        skew={column_feature.get('skewness')} 
        kurtosis={column_feature.get('kurtosis')}
        """ 
    return text.strip()
