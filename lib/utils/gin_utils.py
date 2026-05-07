import gin

def gin_query(s): return gin.query_parameter(f'%{s}')