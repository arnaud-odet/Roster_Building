import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

from roster_lib.utils.stats import find_best_decision_boundary

## Transition plot

def plot_grouped_scatter_with_errorbars(input_df, group_column, x_column, y_column, 
                                        hue_column=None, figsize=(10, 6),
                                        filter_column = None, filter_values = None, 
                                        title=None, cmap='viridis', 
                                        vmin=None, vmax=None):
    """
    Creates a scatter plot with error bars from grouped dataframe data.
    Points, error bars, and labels are colored by a continuous hue column (averaged).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    group_column : str
        Column name to group by (will be used for labels)
    x_column : str
        Column name for x-axis values
    y_column : str
        Column name for y-axis values
    hue_column : str, optional
        Column name for color coding (continuous values, will be averaged)
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title (optional)
    cmap : str
        Colormap name for the continuous hue column
    vmin : float, optional
        Minimum value for color normalization (auto if None)
    vmax : float, optional
        Maximum value for color normalization (auto if None)
    """
    
    df = input_df.copy()
    if filter_column is not None :
        if type(filter_values) is not list :
            filter_values = [filter_values]
        df = df[df[filter_column].isin(filter_values)]
    
    # Group by the specified column and calculate mean and std
    if hue_column:
        # Aggregate both statistics and hue column
        grouped_stats = df.groupby(group_column).agg({
            x_column: ['mean', 'std'],
            y_column: ['mean', 'std'],
            hue_column: ['mean', 'std']  # Average the hue column as well
        })
    else:
        grouped_stats = df.groupby(group_column).agg({
            x_column: ['mean', 'std'],
            y_column: ['mean', 'std']
        })
    
    # Flatten the multi-level column names for easier access
    grouped_stats.columns = ['_'.join(col).strip() for col in grouped_stats.columns.values]
    
    # Reset index to access group column as regular column
    grouped_stats = grouped_stats.reset_index()
    
    # Extract values for plotting
    x_means = grouped_stats[f'{x_column}_mean'].values  # Shape: (n_groups,)
    x_stds = grouped_stats[f'{x_column}_std'].values    # Shape: (n_groups,)
    y_means = grouped_stats[f'{y_column}_mean'].values  # Shape: (n_groups,)
    y_stds = grouped_stats[f'{y_column}_std'].values    # Shape: (n_groups,)
    labels = grouped_stats[group_column].values          # Shape: (n_groups,)
    
    # Handle colors based on hue_column
    if hue_column:
        hue_means = grouped_stats[f'{hue_column}_mean'].values  # Shape: (n_groups,)
        
        # Set up color normalization
        if vmin is None:
            vmin = hue_means.min()
        if vmax is None:
            vmax = hue_means.max()
        
        # Create normalizer and colormap
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.cm.get_cmap(cmap)
        scalar_map = ScalarMappable(norm=norm, cmap=colormap)
        
        # Map hue values to colors
        colors = [colormap(norm(hue_val)) for hue_val in hue_means]  # Shape: (n_groups,)
    else:
        colors = ['steelblue'] * len(labels)
        hue_means = None
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each point individually to control colors
    for i in range(len(x_means)):
        # Get the color for this point
        color = colors[i]
        
        # Plot scatter with error bars
        ax.errorbar(x_means[i], y_means[i], 
                    xerr=x_stds[i],      # Error bars for x-axis
                    yerr=y_stds[i],      # Error bars for y-axis
                    fmt='o',             # Marker style (circles)
                    markersize=6,
                    capsize=3,           # Size of error bar caps
                    capthick=1,
                    elinewidth=0.8,
                    color=color,         # Color for marker and error bars
                    ecolor=color,        # Explicitly set error bar color
                    alpha=0.8,
                    zorder=3)            # Draw on top of grid
        
        # Add labels for each point with matching color (just the label, no hue info)
        label_text = f'{labels[i]}'
        
        ax.annotate(label_text, 
                   (x_means[i], y_means[i]),
                   textcoords="offset points",  # Offset from point
                   xytext=(10, 10),             # Offset in pixels
                   ha='left',
                   fontsize=6,
                   color=color,                 # Text color matches point
                   weight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='white',   # White background for readability
                            alpha=0.8,
                            edgecolor=color,
                            linewidth=2),
                   zorder=4)                    # Draw on top of everything
    
    # Add colorbar for continuous hue if applicable
    if hue_column:
        cbar = plt.colorbar(scalar_map, ax=ax, pad=0.02)
        cbar.set_label(f'{hue_column} (mean)', fontsize=11, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)
    
    # Styling
    ax.set_xlabel(f'{x_column} (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_column} (mean ± std)', fontsize=12, fontweight='bold')
    
    if title:
        plot_title = title
    elif hue_column:
        plot_title = f'{y_column} vs {x_column} by {group_column}\n(colored by {hue_column} mean)'
    else:
        plot_title = f'{y_column} vs {x_column} by {group_column}'
    
    ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    plt.tight_layout()
    return fig, ax


## Display Statsmodel results

def build_display_dict(sm_object, highlight_list:list = None):
    display_dict = {}
    for ind, coeff, p_val in zip(sm_object.params.index, sm_object.params.values, sm_object.pvalues.values) :
        # Display name
        if ind == 'const' or ind == 'Intercept':
            dname = 'Intercept'
        elif type(ind) == int :
            dname = f'Cluster {ind}'
        else :
            dname = ind
        if p_val < 0.05 :
            dname += ' *'
        if p_val < 0.001 :
            dname += '*'
        
        # Display color
        if highlight_list is not None :
            dcol = int(ind != 'const' and ind != 'Intercept') + int(ind in highlight_list)
        else :
            dcol = 0
        
        display_dict[ind] = {'dname': dname, 'dcol': dcol, 'coeff': coeff}
    return display_dict

def plot_regression_coefficients(sm_object, ax= None, highlight_list:list = None, colors:list = ['grey','navy','darkcyan']):
    
    dd = build_display_dict(sm_object=sm_object, highlight_list=highlight_list)
    
    if ax is None :
        fig, ax = plt.subplots(1,1,figsize = (10,6))
        
    ax.bar(x = [item['dname'] for item in dd.values()], 
           height =  [item['coeff'] for item in dd.values()],
           color = [colors[item['dcol']] for item in dd.values()]);
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70);
    
    return ax


## Linear Regression plots 

def simple_linear_regression_plots(target:str, feature:str, data:pd.DataFrame):

    fig, axs = plt.subplots(1,3,figsize = (18,5))
    formula = f"{target} ~ {feature}"
    tmp_linreg = sm.OLS.from_formula(formula, data).fit()
    
    sns.regplot(data = data, x = feature, y = target, ax = axs[0]);
    p_val = tmp_linreg.pvalues.iloc[1]
    str_pval = ('*' if p_val < 0.05 else '') + ('*' if p_val < 0.001 else '') 
    title = f"{feature} | R2 = {tmp_linreg.rsquared_adj:.2f} | coeff = {tmp_linreg.params.iloc[1]:.2f} {str_pval}"
    axs[0].set_title(title)
    
    sns.histplot(tmp_linreg.resid, kde=True, edgecolor='w', ax = axs[1]);
    axs[1].set_title('Residuals Histogram')
    axs[1].set_xlabel('Residuals')

    sns.scatterplot(x = tmp_linreg.predict(data[feature]), y = tmp_linreg.resid, ax = axs[2], alpha = 0.6);
    axs[2].set_title('Residuals vs Predicted Values')
    axs[2].set_xlabel('Predictions')
    axs[2].set_ylabel('Residuals');


## Logistic regression decision boundary

def plot_logistic_decision_boundary_2d(data:pd.DataFrame,
                                    x1: str,
                                    x2: str,
                                    y: str,
                                    scale:bool = True,
                                    hue_col:str = None,
                                    plot_boundaries:bool = True,
                                    show_confusion_matrices:bool = True,
                                    show_summary:bool = True,
                                    step:float = 0.05,
                                    ax=None):
    
    # Step 1 : data handling 
    features = [x1,x2]
    X = data[features]
    Y = data[y].values
    if scale :
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=features)
    X = sm.add_constant(X)
    
    # Step 2 : model & results
    logreg = Logit(Y,X).fit()
    if show_summary:
        print(logreg.summary())
    y_prob =  logreg.predict(X)   
    y_pred_base = (y_prob > 0.5).astype(int)
    acc_base = accuracy_score(Y, y_pred_base)
    f1_base = f1_score(Y, y_pred_base)

    db, f1 = find_best_decision_boundary(y_true=Y, y_prob=y_prob, step = step)
    acc = accuracy_score(Y, (y_prob>db).astype(int))
    if show_confusion_matrices :
        cm_base = confusion_matrix(Y, y_pred_base)
        cm = confusion_matrix(Y, (y_prob>db).astype(int))
        print(f"Base model (decision boudary @ p = 0.5) - Accuracy : {acc_base:.3f}, F1 : {f1_base:.3f}")            
        display(pd.DataFrame(cm_base, index = ['true 0','true 1'], columns = ['pred 0', 'pred 1']))        
        print(f"Adjusted model (decision boudary @ p = {db:.2f}) - Accuracy : {acc:.3f}, F1 : {f1:.3f}")            
        display(pd.DataFrame(cm, index = ['true 0','true 1'], columns = ['pred 0', 'pred 1']))   

    # Step 3 : determine base boundary
    ws = logreg.params[x1], logreg.params[x2] 
    bias = logreg.params['const']
    ### Create decision boundary (where w1*x1_scaled + w2*x2_scaled + b = 0)  
    x_sc_adj = (data[x1].max() - data[x1].min()) / 20
    x_min, x_max = data[x1].min() - x_sc_adj, data[x1].max() + x_sc_adj
    x_range = np.linspace(x_min, x_max, 300)
    x_range_scaled = ((x_range - scaler.mean_[0]) / scaler.scale_[0]) if scale else x_range
    ### Calculate corresponding y values in scaled space (x2_scaled = -(w1*x1_scaled + b) / w2)
    y_range_scaled = -(ws[0] * x_range_scaled + bias) / ws[1]
    y_range = (y_range_scaled * scaler.scale_[1] + scaler.mean_[1]) if scale else y_range_scaled
    # Step 4 : plot
    if ax == None :
        fig, ax = plt.subplots(1,1,figsize = (12,7))
    sns.scatterplot(data = data, 
                    x = x1, 
                    y = x2, 
                    hue = hue_col, 
                    style = 'csf', 
                    markers = {1:'o',0:'X'}, 
                    palette= 'coolwarm',
                    s = 80, 
                    ax = ax);
    y_sc_adj = (data[x2].max() - data[x2].min()) / 20
    y_min, y_max = data[x2].min() - y_sc_adj, data[x2].max() + y_sc_adj
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                        np.linspace(y_min, y_max, 150))
    ### Prepare grid for prediction
    grid = np.c_[xx.ravel(), yy.ravel()] 
    grid_scaled = scaler.transform(grid) if scale else grid  
    grid_scaled = sm.add_constant((grid_scaled))
    probs = logreg.predict(grid_scaled).reshape(xx.shape)
    ### Plot probability contours
    if plot_boundaries:
        contour = ax.contour(xx, yy, probs, 
                            levels=[db,0.5], 
                            colors=['green','green'], 
                            linewidths=[2, 2], 
                            linestyles=['--','-'], 
                            alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=9, fmt='P=%.2f')
    ### Labels and title
    ax.set_xlabel(x1, fontsize=10, fontweight='bold')
    ax.set_ylabel(x2, fontsize=10, fontweight='bold')
    ax.set_title(f"Prediction : @ p=0.5, accuracy = {acc_base:.3f}, f1 = {f1_base:.3f}, @ p={db:.2f}, accuracy = {acc:.3f}, f1 = {f1:.3f}", 
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
