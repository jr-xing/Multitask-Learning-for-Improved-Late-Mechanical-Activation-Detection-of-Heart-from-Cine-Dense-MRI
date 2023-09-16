# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:02:53 2021

@author: Jerry Xing
"""

# def save
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from icecream import ic
from utils.data import get_data_type_by_category, get_data_category_by_type

def plot_strainmat_with_curves(strainMat, curves, curve_types=None, axe=None, legends=None, title=None, vmin=-0.2,
                               vmax=0.2, flipTOS=False, flipStrainMat=False, colors=None):
    # print('plot_strainmat_with_curves')
    if axe is None:
        fig, axe = plt.subplots()

    if strainMat is not None:
        strainMat = np.squeeze(strainMat)
        if flipStrainMat:
            strainMat = np.flip(strainMat, axis=-2)
        nonzero_col_indicator = np.sum(np.abs(strainMat), axis=0) > 1e-3
        axe.pcolor(strainMat[:, :len(nonzero_col_indicator) - np.argmax(nonzero_col_indicator[::-1])], cmap='jet',
                   vmin=vmin, vmax=vmax)
        # axe.pcolor(np.squeeze(strainMat), cmap='jet', vmin = vmin, vmax = vmax)

    if curve_types is None:
        curve_types = ['TOS'] * len(curves)
    if colors is None:
        # colors = ['#0485d1', '#ff9408', 'red', 'blue']
        gt_colors = ['#0485d1', 'blue']
        pred_colors = ['#ff9408', 'red']
    TOS_types = ['TOS', 'TOS18_Jerry', 'TOSInterploated', 'TOSInterpolatedMid', 'TOSfullRes_Jerry']
    TOS_types += ['TOS18', 'TOS126']
    TOS_types += [TOS_type + '_pred' for TOS_type in TOS_types]
    TOS_types += ['activeContourResult', 'activeContourResultFullRes', 'pred_new_multitask', 'pred', 'pred_pred',
                  'pred_new_multitask_masked', 'combined']
    curve_style = {'linewidth': 3}
    for idx, (curve, curve_type) in enumerate(zip(curves, curve_types)):
        if title is not None:
            axe.set_title(title)
            
        if curve is None or curve_type is None:
            continue
        
        curve_type_pure = curve_type.replace('_pred', '')
        # curve_type_pure_category = get_data_type_by_category(curve_type_pure)
        curve_type_pure_category = get_data_category_by_type(curve_type_pure)
        # print(curve_type_pure)
        if curve_type_pure in TOS_types:
            pass
            curve_style = {'linewidth': 3}
        elif curve_type_pure in ['late_acti_label', 'strain_curve_type_label',
                                 'scar-AHA-step', 'scar_sector_label', 'late_activation_sector_label']:
            if curve.shape[-2] == 1:
                curve = np.squeeze(curve > 0.5) * 17 * 3 + 17
            else:
                curve = (np.argmax(curve, axis=1) != 0) * 17 * 3 + 17
            curve_style = {'linewidth': 3}
        elif curve_type_pure_category == 'sector_dist_map':
            pass
        elif curve_type_pure in ['polyfit_coefs']:
            print(curve.flatten())
            xs = np.arange(128)
            curve = np.polyval(curve.flatten(), xs)
            print(type(curve))
        else:
            raise ValueError(f'Unsupported curve type: {curve_type}')
        # curve_grid = np.flip((curve / 17).flatten() - 0.5)
        
        if flipTOS:
            curve = curve[::-1]
            
        if curve_type_pure_category == 'sector_dist_map':
            # Map distance value from [-N_sector, N_sector] to [0, 10]
            curve = curve[0, 0, :]
            curve_norm = 5*(curve+len(curve))/len(curve)
                        
            distmap_threshold = 5
            distmap_in_region_style = {'linewidth': 2, 'linestyle': '-'}
            distmap_out_region_style = {'linewidth': 2, 'linestyle': '--'}
            distmap_threshold_style = {'linewidth': 1, 'linestyle': '--'}
            curve_in_region = np.ones_like(curve_norm) * np.nan; curve_in_region[curve_norm >= distmap_threshold] = curve_norm[curve_norm >= distmap_threshold]
            curve_out_region = np.ones_like(curve_norm) * np.nan; curve_out_region[curve_norm < distmap_threshold] = curve_norm[curve_norm < distmap_threshold]
            line_in_region, = axe.plot(curve_in_region, np.arange(len(curve_in_region)) + 0.5, color=colors[idx], **distmap_in_region_style)
            line_out_region, = axe.plot(curve_out_region, np.arange(len(curve_out_region)) + 0.5, color=colors[idx], **distmap_out_region_style)
            line_threshold,  = axe.plot(np.ones_like(curve_norm)*distmap_threshold, np.arange(len(curve)), color='k', **distmap_threshold_style)
            if legends is not None:
                line_in_region.set_label(legends[idx])
                line_out_region.set_label(legends[idx])
        else:
            curve_grid = (curve / 17).flatten() - 0.5
            if '_pred' in curve_type:
                curr_curve_color = pred_colors[idx//2]
                curve_style['linestyle'] = '--'
            else:
                curr_curve_color = gt_colors[idx//2]
                curve_style['linestyle'] = '-'
            line, = axe.plot(curve_grid, np.arange(len(curve_grid)) + 0.5, color=curr_curve_color, **curve_style)
            if legends is not None:
                line.set_label(legends[idx])
                
        
    if legends is not None:
        axe.legend()

def plot_strainmat_with_curves_OLD(strainMat, curves, curve_types=None, axe=None, legends=None, title=None, vmin=-0.2,
                               vmax=0.2, flipTOS=False, flipStrainMat=False, colors=None):
    # print('plot_strainmat_with_curves')
    if axe is None:
        fig, axe = plt.subplots()

    if strainMat is not None:
        strainMat = np.squeeze(strainMat)
        if flipStrainMat:
            strainMat = np.flip(strainMat, axis=-2)
        nonzero_col_indicator = np.sum(np.abs(strainMat), axis=0) > 1e-3
        axe.pcolor(strainMat[:, :len(nonzero_col_indicator) - np.argmax(nonzero_col_indicator[::-1])], cmap='jet',
                   vmin=vmin, vmax=vmax)
        # axe.pcolor(np.squeeze(strainMat), cmap='jet', vmin = vmin, vmax = vmax)

    if curve_types is None:
        curve_types = ['TOS'] * len(curves)
    if colors is None:
        colors = ['#0485d1', '#ff9408', 'red', 'blue']
    TOS_types = ['TOS', 'TOS18_Jerry', 'TOSInterploated', 'TOSInterpolatedMid', 'TOSfullRes_Jerry']
    TOS_types += ['TOS18', 'TOS126']
    TOS_types += [TOS_type + '_pred' for TOS_type in TOS_types]
    TOS_types += ['activeContourResult', 'activeContourResultFullRes', 'pred_new_multitask', 'pred', 'pred_pred',
                  'pred_new_multitask_masked']
    curve_style = {'linewidth': 4}
    for idx, (curve, curve_type) in enumerate(zip(curves, curve_types)):
        # ic(curve)
        # print(curve.shape)
        # return
        if title is not None:
            axe.set_title(title)
            
        if curve is None or curve_type is None:
            continue
        
        # print(curve, curve_type)
        # print(curve)
        curve_type_pure = curve_type.replace('_pred', '')
        # curve_type_pure_category = get_data_type_by_category(curve_type_pure)
        curve_type_pure_category = get_data_category_by_type(curve_type_pure)
        # print(curve_type_pure)
        if curve_type_pure in TOS_types:
            pass
            curve_style = {'linewidth': 4}
        elif curve_type_pure in ['late_acti_label', 'strain_curve_type_label',
                                 'scar-AHA-step', 'scar_sector_label', 'late_activation_sector_label']:
            # curve = curve*17*3 + 17
            # print(curve.shape)
            if curve.shape[-2] == 1:
                # print(curve)
                curve = np.squeeze(curve > 0.5) * 17 * 3 + 17
            else:
                curve = (np.argmax(curve, axis=1) != 0) * 17 * 3 + 17
            # curve = (curve > 0.5) * 17 * 3 + 17
            # print(curve)
            curve_style = {'linewidth': 2, 'linestyle': '--'}
            # if curve_type == 'late_acti_label_pred':
            #     print(curve)
        # elif curve_type_pure in ['late_acti_dist_map', 'strain_curve_type_dist_map',
        #                          'scar-AHA-distmap', 'scar_sector_precentage','scar_sector_distmap']:
        #     curve = curve[0, 0, :]            
        #     # curve = (curve > 0) * 17*3 + 17
        #     # curve = (curve) * 20 + 17
        #     # Normalize distmap into [0,1]
        #     # curve = np.abs(curve)
        #     # curve = (curve - np.min(curve)) / (np.max(curve) - np.min(curve))
        #     curve = (curve) * 5 + 17
        #     # curve[curve<0] = 0
        #     curve_style = {'linewidth': 2, 'linestyle': '--'}
        elif curve_type_pure_category == 'sector_dist_map':
            pass
        elif curve_type_pure in ['polyfit_coefs']:
            print(curve.flatten())
            xs = np.arange(128)
            curve = np.polyval(curve.flatten(), xs)
            print(type(curve))
        else:
            raise ValueError(f'Unsupported curve type: {curve_type}')
        # curve_grid = np.flip((curve / 17).flatten() - 0.5)
        
        if flipTOS:
            curve = curve[::-1]
            
        if curve_type_pure_category == 'sector_dist_map':
            # Map distance value from [-N_sector, N_sector] to [0, 10]
            curve = curve[0, 0, :]
            curve_norm = 5*(curve+len(curve))/len(curve)
                        
            distmap_threshold = 5
            distmap_in_region_style = {'linewidth': 2, 'linestyle': '-'}
            distmap_out_region_style = {'linewidth': 2, 'linestyle': '--'}
            distmap_threshold_style = {'linewidth': 1, 'linestyle': '--'}
            curve_in_region = np.ones_like(curve_norm) * np.nan; curve_in_region[curve_norm >= distmap_threshold] = curve_norm[curve_norm >= distmap_threshold]
            curve_out_region = np.ones_like(curve_norm) * np.nan; curve_out_region[curve_norm < distmap_threshold] = curve_norm[curve_norm < distmap_threshold]
            line_in_region, = axe.plot(curve_in_region, np.arange(len(curve_in_region)) + 0.5, color=colors[idx], **distmap_in_region_style)
            line_out_region, = axe.plot(curve_out_region, np.arange(len(curve_out_region)) + 0.5, color=colors[idx], **distmap_out_region_style)
            line_threshold,  = axe.plot(np.ones_like(curve_norm)*distmap_threshold, np.arange(len(curve)), color='k', **distmap_threshold_style)
            if legends is not None:
                line_in_region.set_label(legends[idx])
                line_out_region.set_label(legends[idx])
        else:
            curve_grid = (curve / 17).flatten() - 0.5
            line, = axe.plot(curve_grid, np.arange(len(curve_grid)) + 0.5, color=colors[idx], **curve_style)
            if legends is not None:
                line.set_label(legends[idx])
                
        
    if legends is not None:
        axe.legend()

    # return fig, axe


def save_strainmat_with_curves(strainMat, curves, save_filename, curve_types=None, legends=None, title=None, vmin=-0.2,
                               vmax=0.2, flipTOS=False, flipStrainMat=False, colors=None):
    plt.ioff()
    fig, axe = plt.subplots()

    plot_strainmat_with_curves(strainMat, curves, curve_types,
                               axe=axe, legends=legends,
                               title=title,
                               vmin=vmin, vmax=vmax,
                               flipTOS=flipTOS, flipStrainMat=flipStrainMat)
    fig.savefig(save_filename, bbox_inches='tight')  # save the figure to file
    plt.close(fig)
    plt.ion()


def plot_multi_strainmat_with_curves(data: list, strainmat_type: str, curve_types: list, \
                                     fig=None, axs=None, \
                                     legends=None, title=None, subtitles=None, \
                                     vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
                                     n_cols=4, colors=None):
    n_rows = len(data) // n_cols + 1 if len(data) % n_cols != 0 else len(data) // n_cols
    if axs is None:
        figHeight = n_rows * 4
        figWidth = n_cols * 5
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(figWidth, figHeight))
        axs = axs.flatten()
    if subtitles is None:
        subtitles = [None] * len(data)

    if len(data) > len(axs):
        raise ValueError(f'# of data ({len(data)}) should not be larger than # of axs ({len(axs)})')

    for idx, datum in enumerate(data):
        plot_strainmat_with_curves(
            strainMat=datum[strainmat_type],
            curves=[datum[curve_type] for curve_type in curve_types],
            curve_types=curve_types,
            axe=axs[idx],
            legends=legends,
            title=subtitles[idx],
            vmin=vmin, vmax=vmax,
            flipTOS=flipTOS, flipStrainMat=flipStrainMat, colors=colors)
    # print(len(data), n_rows * n_cols)
    # for idx_blank in range(len(data), n_rows * n_cols):
    #     axs[idx_blank].set_axis_off()
    #     axs[idx_blank].axis('off')
    #     print(idx_blank)

    if title is not None:
        fig.suptitle(title, y=0.02)


def save_multi_strainmat_with_curves(data: list, strainmat_type: str, curve_types: list, \
                                     save_filename, \
                                     fig=None, axs=None, \
                                     legends=None, title=None, subtitles=None, \
                                     vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
                                     n_cols=4,
                                     enable_multipages=True, n_rows_per_page=4, colors=None):
    if subtitles is None:
        subtitles = [None] * len(data)

    plt.ioff()
    if enable_multipages is False or n_cols * n_rows_per_page >= len(data):
        n_rows = len(data) // n_cols + 1 if len(data) % n_cols != 0 else len(data) // n_cols
        figHeight = n_rows * 4
        figWidth = n_cols * 5
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(figWidth, figHeight))
        if n_cols*n_rows_per_page > 1:
            axs = axs.flatten()
        plot_multi_strainmat_with_curves(data, strainmat_type, curve_types,
                                         fig, axs,
                                         legends, title, subtitles,
                                         vmin, vmax, flipTOS, flipStrainMat,
                                         n_cols)
        fig.savefig(save_filename, bbox_inches='tight')  # save the figure to file
        plt.close(fig)
    else:
        assert save_filename.split('.')[-1].lower() == 'pdf', f'Should save as .pdf file instead of {save_filename.split(".")[-1]}!'
        n_plots_per_page = n_cols * n_rows_per_page
        n_pages = len(data) // n_plots_per_page + 1 if len(data) % n_plots_per_page != 0 else len(
            data) // n_plots_per_page
        with PdfPages(save_filename) as pdf:
            figHeight = n_rows_per_page * 4
            figWidth = n_cols * 5
            for page_idx in range(n_pages):
                fig, axs = plt.subplots(n_rows_per_page, n_cols, figsize=(figWidth, figHeight))
                if n_plots_per_page > 1:
                    axs = axs.flatten()
                data_starts_idx = page_idx * n_plots_per_page
                data_ends_idx = data_starts_idx + n_plots_per_page
                data_to_plot = data[data_starts_idx:data_ends_idx]
                plot_multi_strainmat_with_curves(data_to_plot, strainmat_type, curve_types,
                                                 fig, axs,
                                                 legends, title, subtitles[data_starts_idx:data_ends_idx],
                                                 vmin, vmax, flipTOS, flipStrainMat,
                                                 n_cols)
                for idx in range(len(data_to_plot), n_cols * n_rows_per_page):
                    axs[idx].axis('off')
                pdf.savefig(fig, bbox_inches='tight', dpi=50)  # save the figure to file
                plt.close(fig)
    plt.ion()

def plot_activation_map(activation_map, strainmat=None, axe=None, 
                        hide_zero_area=True, 
                        overlap = True, background_color='white'):    
    # Create new fig if not provided
    if axe is None:
        fig, axe = plt.subplots()
    
    # Plot strain matrix
    if strainmat is None:
        strainmat = np.ones_like(activation_map)
        strainmat_vmax = 1
        strainmat_vmin = 0
    else:
        strainmat_vmax = 0.2
        strainmat_vmin = -0.2    
        
    strainmat_squeeze = np.squeeze(strainmat)
    
    # axe.pcolor(np.squeeze(strainmat), vmax = strainmat_vmax, vmin = strainmat_vmin, cmap = 'jet')
    
    if hide_zero_area:
        nonzero_col_indicator = np.sum(np.abs(strainmat_squeeze), axis=0) > 1e-3
        last_nonzero_frame = len(nonzero_col_indicator) - np.argmax(nonzero_col_indicator[::-1])
    else:
        last_nonzero_frame = strainmat_squeeze.shape[-1]
    
    if overlap:
        axe.pcolor(strainmat_squeeze[:, :last_nonzero_frame], vmax = strainmat_vmax, vmin = strainmat_vmin, cmap = 'jet')
    else:
        background_color = 'black'
    
    # Plot activation map
    activation_map_squeeze = np.squeeze(activation_map)
    if np.max(activation_map) > 1e-5:
        activation_map_norm = (activation_map_squeeze - np.min(activation_map_squeeze)) / (np.max(activation_map_squeeze) - np.min(activation_map_squeeze))
        alpha_threshold = 0.9
    else:
        activation_map_norm = activation_map_squeeze
        alpha_threshold = 1.0

    activation_map_norm_crop = activation_map_norm[:, :last_nonzero_frame]
    # axe.pcolor(np.zeros_like(strainmat_squeeze[:, :last_nonzero_frame]), alpha=np.maximum(1-activation_map_norm_crop, 0.5), cmap='gray', vmin = 0, vmax = 1)
    if background_color == 'white':
        axe.pcolor(np.ones_like(strainmat_squeeze[:, :last_nonzero_frame]), alpha=np.minimum(1-activation_map_norm_crop, 1.0), cmap='gray', vmin = 0, vmax = 1)
    elif background_color == 'black':
        axe.pcolor(np.zeros_like(strainmat_squeeze[:, :last_nonzero_frame]), alpha=np.minimum(1-activation_map_norm_crop, alpha_threshold), cmap='gray', vmin = 0, vmax = 1)
    elif background_color == 'gray':
        axe.pcolor(np.ones_like(strainmat_squeeze[:, :last_nonzero_frame])*0.7, alpha=np.minimum(1-activation_map_norm_crop, 1.0), cmap='gray', vmin = 0, vmax = 1)
    # axe.pcolor(activation_map_norm_crop, cmap='gray', vmin = 0, vmax = 1)
    

def plot_multi_strainmat_with_curves_and_activation_map(data: list, strainmat_type: str, curve_types: list, \
                                     cam_data_types:list=None, counter_cam_data_types:list=None, \
                                     fig=None, axs=None, \
                                     n_cols=None,\
                                     legends=None, title=None, subtitles=None, \
                                     vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
                                     overlap = True, \
                                     colors=None, check_activation_sectors=None,background_color='gray'):
    n_rows = len(data)    
    # print(n_cols)
    # print(cam_data_types, counter_cam_data_types)
    if axs is None:
        n_cols = len(cam_data_types) + len(counter_cam_data_types) + 1
        figHeight = n_rows * 4
        figWidth = n_cols * 5
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(figWidth, figHeight))
    # ic(n_cols)
    # ic(len(cam_data_types) + len(counter_cam_data_types) + 1)
    axs = axs.flatten()
    if subtitles is None:
        subtitles = [None] * len(data)

    # if len(data) > len(axs):
    #     raise ValueError(f'# of data ({len(data)}) should not be larger than # of axs ({len(axs)})')
    # ic(len(data))
    if check_activation_sectors is None:
        check_activation_sectors = [-1] * len(data)
    # For each datum
    for datum_idx, datum in enumerate(data):
        row_axe_idx_start = datum_idx * n_cols
        
        # 1. Plot strain matrix and prediction
        plot_strainmat_with_curves(
            strainMat=datum[strainmat_type],
            curves=[datum[curve_type] for curve_type in curve_types] if curve_types is not None else [None]*len(data),
            curve_types=curve_types,
            axe=axs[row_axe_idx_start],
            legends=legends,
            title=subtitles[datum_idx],
            vmin=vmin, vmax=vmax,
            flipTOS=flipTOS, flipStrainMat=flipStrainMat, colors=colors)
        # axs[row_axe_idx_start].set_title('Strain Matrix')
                
        # 2. Plot cam for all data types
        unique_cam_data_types = np.unique(cam_data_types + counter_cam_data_types)
        # ic(unique_cam_data_types)
        # return
        axe_idx = 1
        for cam_data_type_idx, cam_data_type in enumerate(unique_cam_data_types):            
            if cam_data_type in cam_data_types:
                plot_activation_map(datum['cam'][cam_data_type]['cam'], datum[strainmat_type], axs[row_axe_idx_start + axe_idx], overlap = overlap, background_color=background_color)
                if 'loss' not in cam_data_type:
                    sub_title = f'{cam_data_type} CAM at sector {check_activation_sectors[datum_idx]}'
                else:
                    sub_title = f'{cam_data_type} CAM'
                axs[row_axe_idx_start + axe_idx].set_title(sub_title)
                axe_idx += 1
            if cam_data_type in counter_cam_data_types:
                plot_activation_map(datum['cam'][cam_data_type]['counter_cam'], datum[strainmat_type], axs[row_axe_idx_start + axe_idx], overlap = overlap, background_color=background_color)
                if 'loss' not in cam_data_type:
                    sub_title = f'{cam_data_type} Counter CAM at sector {check_activation_sectors[datum_idx]}'
                else:
                    sub_title = f'{cam_data_type} Counter CAM'
                axs[row_axe_idx_start + axe_idx].set_title(sub_title)
                axe_idx += 1
        
    # print(len(data), n_rows * n_cols)
    # for idx_blank in range(len(data), n_rows * n_cols):
    #     axs[idx_blank].set_axis_off()
    #     axs[idx_blank].axis('off')
    #     print(idx_blank)

    if title is not None:
        fig.suptitle(title, y=0.02)
    
    return fig

def save_multi_strainmat_with_curves_and_activation_map(data: list, strainmat_type: str, curve_types: list, \
                                     save_filename, \
                                     cam_data_types:list=None, counter_cam_data_types:list=None, \
                                     activation_map_type = 'activation_map', counter_activation_map_type = 'counter_activation_map', \
                                     fig=None, axs=None, \
                                     legends=None, title=None, subtitles=None, \
                                     vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
                                     colors=None, overlap=True, check_activation_sectors=None, background_color='gray',\
                                     rasterization=True):
    if subtitles is None:
        subtitles = [None] * len(data)

    plt.ioff()
    file_extension = save_filename.split('.')[-1].lower()
    if file_extension == 'pdf':
        n_rows_per_page = 1
        n_cols = len(cam_data_types) + len(counter_cam_data_types) + 1
        n_plots_per_page = n_rows_per_page
        n_pages = len(data)
        with PdfPages(save_filename) as pdf:
            figHeight = n_rows_per_page * 4
            figWidth = n_cols * 5
            for page_idx in range(n_pages):
                # Create figure
                # ic(n_cols)
                fig, axs = plt.subplots(1, n_cols, figsize=(figWidth, figHeight))
                axs = axs.flatten()
                
                # Get data to plot
                data_starts_idx = page_idx * n_plots_per_page
                data_ends_idx = data_starts_idx + n_plots_per_page
                data_to_plot = data[data_starts_idx:data_ends_idx]#[0]
                data_target_sector_idx = check_activation_sectors[data_starts_idx:data_ends_idx]
                # print(data_starts_idx, data_ends_idx)
                # print(len(data_to_plot))
                # print(type(data_to_plot))
                
                # Plot
                plot_multi_strainmat_with_curves_and_activation_map(
                    data=data_to_plot, strainmat_type=strainmat_type, curve_types=curve_types, \
                    cam_data_types=cam_data_types, counter_cam_data_types=counter_cam_data_types, \
                    fig=fig, axs=axs, n_cols=n_cols,\
                    legends=legends, title=title, subtitles=subtitles[data_starts_idx:data_ends_idx], \
                    vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
                    colors=colors, overlap=overlap, check_activation_sectors=data_target_sector_idx, background_color=background_color)                        
                
                # for idx in range(len(data_to_plot), n_cols * n_rows_per_page):
                #     axs[idx].axis('off')
                if rasterization:
                    for axe in axs:
                        axe.set_rasterization_zorder(0)
                pdf.savefig(fig, bbox_inches='tight', dpi=10)  # save the figure to file
                plt.close(fig)
    else:
        n_rows_per_img = 4
        n_cols = len(cam_data_types) + len(counter_cam_data_types) + 1
        n_imgs = np.ceil(len(data)/n_rows_per_img).astype(int)
        
        figHeight = n_rows_per_img * 4
        figWidth = n_cols * 5
        for img_idx in range(n_imgs):
            if n_imgs > 1:
                curr_save_filename = save_filename.replace('.'+file_extension, f'-{img_idx}.'+file_extension)
            else:
                curr_save_filename = save_filename
            
            # Create figure
            # ic(n_cols)
            fig, axs = plt.subplots(n_rows_per_img, n_cols, figsize=(figWidth, figHeight))
            axs = axs.flatten()
            
            # Get data to plot
            data_starts_idx = img_idx * n_rows_per_img
            data_ends_idx = data_starts_idx + n_rows_per_img
            data_to_plot = data[data_starts_idx:data_ends_idx]#[0]
            data_target_sector_idx = check_activation_sectors[data_starts_idx:data_ends_idx]
            
            # Plot
            plot_multi_strainmat_with_curves_and_activation_map(
                data=data_to_plot, strainmat_type=strainmat_type, curve_types=curve_types, \
                cam_data_types=cam_data_types, counter_cam_data_types=counter_cam_data_types, \
                fig=fig, axs=axs, n_cols=n_cols,\
                legends=legends, title=title, subtitles=subtitles[data_starts_idx:data_ends_idx], \
                vmin=-0.2, vmax=0.2, flipTOS=False, flipStrainMat=False, \
                colors=colors, overlap=overlap, check_activation_sectors=data_target_sector_idx, background_color=background_color)                        
            
            # for idx in range(len(data_to_plot), n_cols * n_rows_per_page):
            #     axs[idx].axis('off')
            # plt.savefig(fig, bbox_inches='tight')  # save the figure to file
            print(curr_save_filename)
            fig.savefig(curr_save_filename, bbox_inches='tight')  # save the figure to file
            plt.close(fig)
    plt.ion()
