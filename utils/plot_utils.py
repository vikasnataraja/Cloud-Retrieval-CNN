import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from datetime import datetime
now = datetime.now().strftime('%b-%d-%Y_%I-%M-%p')

def plot_training(history, modelname):
    """ plot training and validation losses over epochs """
    modelname = os.path.splitext('.')[0]
    fig = plt.figure(figsize=(20,16))

    plt.plot(history.history['loss'], c='blue', label='Training')
    plt.plot(history.history['val_loss'], c='orange', label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    if not os.path.isdir('results/'):
        os.makedirs('results/')
    fig.savefig('results/{}_training_history_{}.png'.format(modelname, now), dpi=100)
    print('Saved figure in results/{}_training_history_{}'.format(modelname, now))
    plt.close();


def prediction_panel_viz(radiance, cot_true, cot_1d, cot_cnn):

    fig = plt.figure(figsize=(100, 80))
    gs = GridSpec(1, 5, figure=fig)

    # normalization and axis limits
    stop = np.max((cot_true.ravel(), cot_1d.ravel(), cot_cnn.ravel()))
    norm_colors = matplotlib.colors.Normalize(vmin=0., vmax=stop)
    ticks = [0, 20, 40, 60]
    ticklabels = [0, 2, 4, 6]
    cbar_ticks = np.linspace(0, np.round(stop,-1), 5, dtype='int')

    # radiance
    ax = fig.add_subplot(gs[0])
    ax.tick_params(direction='out', length=20, width=2)
    ax.imshow(radiance, cmap='jet', extent=[0, 64, 0, 64], alpha=1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=78, fontweight="bold")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=78, fontweight="bold")
    ax.set_xlabel('X [km]', fontsize=84, fontweight="bold", labelpad=20)
    ax.set_ylabel('Y [km]', fontsize=84, fontweight="bold", labelpad=20)
    ax.set_title('Radiance (600 nm)', fontsize=80, fontweight="bold", pad=30)
    ax.text(0.05, 0.85, 'a)', fontsize=84, fontweight="bold", color='white', transform=ax.transAxes)

    # true COT
    ax = fig.add_subplot(gs[1])
    ax.tick_params(direction='out', length=20, width=2)
    y1 = ax.imshow(cot_true, cmap='jet', extent=[0, 64, 0, 64], norm=norm_colors)
    ax.set_title('True COT', fontsize=80, fontweight="bold", pad=30)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=84, fontweight="bold")
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels, fontsize=84, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel('X [km]', fontsize=84, fontweight="bold", labelpad=20)
    # ax.set_ylabel('Y [km]', fontsize=70, fontweight="bold", labelpad=20)
    ax.text(0.05, 0.85, 'b)', fontsize=84, fontweight="bold", color='white', transform=ax.transAxes)
    # tickbounds = np.linspace(0,35,8)
    # textbounds = pxvals

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.25)
    cb = fig.colorbar(y1, cax=cax, ax=ax, ticks=cbar_ticks)
    cb.ax.tick_params(labelsize=64)

    # IPA COT
    ax = fig.add_subplot(gs[2])
    ax.tick_params(direction='out', length=20, width=2)
    y2 = ax.imshow(cot_1d, cmap='jet', extent=[0, 64, 0, 64], norm=norm_colors)
    ax.set_title('IPA COT', fontsize=80, fontweight="bold", color='green', pad=30)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=84, fontweight="bold")
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels, fontsize=84, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel('X [km]', fontsize=84, fontweight="bold", labelpad=20)
    # ax.set_ylabel('Y [km]', fontsize=70, fontweight="bold", labelpad=20)
    ax.text(0.05, 0.85, 'c)', fontsize=84, fontweight="bold", color='white', transform=ax.transAxes)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.25)
    cb = fig.colorbar(y2, cax=cax, ax=ax, ticks=cbar_ticks)
    cb.ax.tick_params(labelsize=64)

    # CNN COT
    ax = fig.add_subplot(gs[3])
    ax.tick_params(direction='out', length=20, width=2)
    y3 = ax.imshow(cot_cnn, cmap='jet', extent=[0, 64, 0, 64], norm=norm_colors)
    ax.set_title('CNN COT', fontsize=80, fontweight="bold", color='crimson', pad=30)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=84, fontweight="bold")
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticklabels, fontsize=84, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel('X [km]', fontsize=84, fontweight="bold", labelpad=20)
    # ax.set_ylabel('Y [km]', fontsize=70, fontweight="bold", labelpad=20)
    ax.text(0.05, 0.85, 'd)', fontsize=84, fontweight="bold", color='white', transform=ax.transAxes)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.25)
    cb = fig.colorbar(y3, cax=cax, ax=ax, ticks=cbar_ticks)
    cb.ax.tick_params(labelsize=64)

    # scatter plot
    ax = fig.add_subplot(gs[4])
    ax.tick_params(direction='out', length=20, width=2)
    ax.scatter(cot_true.ravel(), cot_1d.ravel()-cot_true.ravel(), s=300, color='green', alpha=0.6)
    ax.scatter(cot_true.ravel(), cot_cnn.ravel()-cot_true.ravel(), s=300, color='crimson', alpha=0.6)
    ax.plot([0, 60], [0, 0], color='black', linestyle='--', linewidth=7)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xticklabels([0, 5, 10, 15], fontsize=84, fontweight="bold")
    ax.set_yticks([-10, -5, 0, 5])
    ax.set_yticklabels([-10, -5, 0, 5], fontsize=84, fontweight="bold")
    ax.set_xlim([-2, 17])
    ax.set_ylim([-12, 7])
    ax.set_xlabel('True COT', fontsize=84, fontweight="bold", labelpad=20)
    ax.set_ylabel('(Retrieved - True) COT', fontsize=70, fontweight="bold", labelpad=25)
    ax.yaxis.set_label_position("right")
    ax.set_aspect("equal")
    ax.text(0.05, 0.2, 'IPA COT', fontsize=72, fontweight="bold", color='green', transform=ax.transAxes)
    ax.text(0.05, 0.1, 'CNN COT', fontsize=66, fontweight="bold", color='crimson', transform=ax.transAxes)
    ax.text(0.05, 0.85, 'e)', fontsize=84, fontweight="bold", color='black', transform=ax.transAxes)

    gs.update(wspace=0.35)
    fig.savefig('cnn_cot_prediction_{}.png'.format(now), dpi=100, bbox_inches = 'tight', pad_inches = 0.25)
    # plt.show()
    plt.close();
