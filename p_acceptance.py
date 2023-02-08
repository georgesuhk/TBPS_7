# coded by Henry

import statistics as stat
import json

#polynomials to fit
n6_polynomial = lambda x, a,b,c,d,e,f,g : a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x*2 + f*x + g
n6_polynomial_even = lambda x, a,b,c,d : a*x**6 + b*x**4 + c*x**2 + d

def get_acceptance(variable: str, value: float):
    """This is the function that outputs the acceptance given angle
    Args:
        variable(str): the angular variable we are looking for, please input either 'l', 'k', or 'phi'
        value(float): the value of that variable
    Returns:
        acceptance with statistical error
    """
    with open('acceptance_'+variable+'_params.json') as file_:
        params = json.load(file_)
    raw_data_params = params.pop()
    #need to handle k seperately since it uses the even polynomial function
    if variable == 'k'
        acceptance = 1/n6_polynomial(value, raw_data_params)
        efficiencies = [n6_polynomial(value, *param) for param in params]
    else:
        acceptance = 1/n6_polynomial_even(value, raw_data_params)
        efficiencies = [n6_polynomial_even(value, *param) for param in params]
    error = stats.stdev(efficiencies)

    return acceptance, error
