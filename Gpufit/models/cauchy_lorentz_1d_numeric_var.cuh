#ifndef GPUFIT_CAUCHY_LORENTZ_1D_NUM_VAR_CUH_INCLUDED
#define GPUFIT_CAUCHY_LORENTZ_1D_NUM_VAR_CUH_INCLUDED

/* Description of the calculate_cauchy_lorentz_1d function
* ===================================================
* Added by Sebastian Hambura the 06/2020
*
* This function calculates the values of the Cauchy distribution model functions
* (also known as Lorentz distribution, Cauchy�Lorentz distribution or Lorentz(ian) function)
* and their partial derivatives with respect to the model parameters.
* https://en.wikipedia.org/wiki/Cauchy_distribution
* The function isn't expected to be normalized nor of 0-offset : f(x) =  A / (1 + [(x-x0)/gamma]**2) + offset
*
* This function makes use of the user information data to pass in the
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* Note that if no user information is provided, the (X) coordinate of the
* first data value is assumed to be (0.0).  In this case, for a fit size of
* M data points, the (X) coordinates of the data are simply the corresponding
* array index values of the data array, starting from zero.
*
* There are three possibilities regarding the X values:
*
*   No X values provided:
*
*       If no user information is provided, the (X) coordinate of the
*       first data value is assumed to be (0.0).  In this case, for a
*       fit size of M data points, the (X) coordinates of the data are
*       simply the corresponding array index values of the data array,
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then
*       the same X values will be used for all fits.  In this case, the
*       size of the user_info array (in bytes) must equal
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters 
*             p[0]: amplitude 
*             p[1]: x0 
*             p[2]: gamma
*             p[3]: offset
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_linear1d function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/


__device__ REAL get_value_lor_var (
	///////////// value calculated here to be used for value and derivative below /////////////
	REAL const* parameters,
    int const n_parameters, 
	int const point_index,
	REAL const x
    )
{
    
    REAL m = parameters[0];
    REAL b = parameters[1];

    REAL x0;
    REAL gamma;
    REAL A;
    REAL denominator;

    int i=2;

    REAL value = x * m + b;

    for(i=2; i<n_parameters; i+=3){
        A = parameters[i];
        x0 = parameters[i+1];
        gamma = parameters[i+2];
        
        denominator = gamma * gamma + (x - x0) * (x - x0);
        value += A * gamma * gamma / denominator;
        
    }

	return  value;
}


__device__ REAL get_der_lor_var (
	///////////// value calculated here to be used for value and derivative below /////////////
	REAL const* parameters,
    int const n_parameters, 
	int const point_index,
    int const derivative_index,
    REAL delta,
	REAL const x
    )
{
    
    REAL m = derivative_index == 0 ? parameters[0] + delta : parameters[0];
    REAL b = derivative_index == 0 ? parameters[1] + delta : parameters[1];

    REAL x0;
    REAL gamma;
    REAL A;
    REAL denominator;

    int i=2;

    REAL value1 = x * m + b;

    for(i=2; i<n_parameters; i+=3){
        A = derivative_index == i ? parameters[i]+ delta : parameters[i];
        x0 = derivative_index == i ? parameters[i+1]+ delta : parameters[i+1];
        gamma = derivative_index == i ? parameters[i+2]+ delta : parameters[i+2];
        
        denominator = gamma * gamma + (x - x0) * (x - x0);
        value1 += A * gamma * gamma / denominator;  
    }

    m = derivative_index == 0 ? parameters[0] - delta : parameters[0];
    b = derivative_index == 0 ? parameters[1] - delta : parameters[1];

    REAL value2 = x * m + b;

    for(i=2; i<n_parameters; i+=3){
        A = derivative_index == i ? parameters[i] - delta : parameters[i];
        x0 = derivative_index == i ? parameters[i+1] - delta : parameters[i+1];
        gamma = derivative_index == i ? parameters[i+2] - delta : parameters[i+2];
        
        denominator = gamma * gamma + (x - x0) * (x - x0);
        value2 += A * gamma * gamma / denominator;  
    }

    REAL value = 1/(2*delta)*(value1-value2);

	return  value;
}



__device__ void calculate_cauchy_lorentz_1d_num_var(
    REAL const* parameters,
    int const n_fits,
    int const n_points,
    REAL* value,
    REAL* derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char* user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL* user_info_float = (REAL*)user_info;
    float n_paramsF = static_cast<float>(*user_info);
    int n_params = static_cast<int>(n_paramsF);

    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if ((user_info_size - sizeof(REAL)) / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index+1];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index + 1];
    }

    // parameters
    
    value[point_index] = get_value_lor_var(parameters, n_params, point_index, x);

    // derivatives
    // REAL squarred_denominator = denominator * denominator;
    REAL* current_derivatives = derivative + point_index;

    // current_derivatives[0 * n_points] = gamma * gamma / denominator;                                // derivative A
    // current_derivatives[1 * n_points] = 2 * A * gamma * gamma * (x - x0) / squarred_denominator;    // derivative x0
    // current_derivatives[2 * n_points] = 2 * A * gamma * (x - x0) * (x - x0) / squarred_denominator; // derivative gamma
    // current_derivatives[3 * n_points] = 1;

    REAL delta = 1e-3;
    
    int idx=0;

    for (idx=0; idx<n_params; idx+=1){
        current_derivatives[idx * n_points] = get_der_lor_var(parameters, n_params, point_index, idx, delta, x);
    }

    
    

    // f_plus_h = get_value_lor(A + h, x0, gamma, offset, point_index, x);
    // f_minus_h = get_value_lor(A - h, x0, gamma, offset, point_index, x);
    // current_derivatives[0 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h); //d/dA

    // h = x0 * delta;

    // f_plus_h = get_value_lor(A, x0+h, gamma, offset, point_index, x);
    // f_minus_h = get_value_lor(A, x0-h, gamma, offset, point_index, x);
    // current_derivatives[1 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h); //d/dx0

    // h = gamma * delta;

    // f_plus_h = get_value_lor(A, x0, gamma+h, offset, point_index, x);
    // f_minus_h = get_value_lor(A, x0, gamma-h, offset, point_index, x);
                                   
    // current_derivatives[2 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h); // derivative d/dgamma

    // h = offset * delta;

    // f_plus_h = get_value_lor(A, x0, gamma, offset+h, point_index, x);
    // f_minus_h = get_value_lor(A, x0, gamma, offset-h, point_index, x);
    // current_derivatives[3 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h); // d/doffset


}

#endif
