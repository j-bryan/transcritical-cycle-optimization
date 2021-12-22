import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.widgets import Slider, RadioButtons


RADIO_LABELS = {
    'Eta': 'eta_FL',
    'Penalty': 'penalty',
    'LCOE': 'LCOE'
}


def get_norm(label, cmin, cmax):
    if label == 'LCOE' and np.log10(cmax) > 6:
        return LogNorm(cmin, cmax * 1e-3)
    else:
        return Normalize(cmin, cmax)


def plot_app(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.35)

    radlabel = 'Eta'
    cmin = min(data[RADIO_LABELS.get(radlabel)])
    cmax = max(data[RADIO_LABELS.get(radlabel)])

    pmaxi = 8.2
    f1i = 1/6
    f2i = 1/6
    f3i = 1/6

    good_subset = data.loc[(np.isclose(data['P_max'], pmaxi - 8.2))
                           & np.isclose(data['f1'], f1i)
                           & np.isclose(data['f2'], f2i)
                           & np.isclose(data['f3'], f3i)]

    l = ax.scatter(good_subset['Pr_1'], good_subset['Pr_2'], good_subset['Pr_3'],
                   c=good_subset[RADIO_LABELS.get(radlabel)], vmin=cmin, vmax=cmax, cmap='jet')
    cb = fig.colorbar(l, label=radlabel)

    ax.set_xlabel('Pr_1')
    ax.set_ylabel('Pr_2')
    ax.set_zlabel('Pr_3')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    axcolor = 'lightgoldenrodyellow'
    axpmax = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor=axcolor)
    axf1 = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
    axf2 = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
    axf3 = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)

    spmax = Slider(axpmax, 'P_max', 8.2, 9.2, valinit=8.2, valstep=0.25)
    sf1 = Slider(axf1, 'f1', 1/6, 5/6, valinit=1/6, valstep=1/6)
    sf2 = Slider(axf2, 'f2', 1/6, 5/6, valinit=1/6, valstep=1/6)
    sf3 = Slider(axf3, 'f3', 1/6, 5/6, valinit=1/6, valstep=1/6)

    rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
    radio = RadioButtons(rax, ('Eta', 'Penalty', 'LCOE'), active=0)

    def update(val):
        ax.clear()

        radlabel = radio.value_selected
        cmin = min(data[RADIO_LABELS.get(radlabel)])
        cmax = max(data[RADIO_LABELS.get(radlabel)])
        norm = get_norm(radlabel, cmin, cmax)

        ax.set_xlabel('Pr_1')
        ax.set_ylabel('Pr_2')
        ax.set_zlabel('Pr_3')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        pmax = spmax.val
        f1 = sf1.val
        f2 = sf2.val
        f3 = sf3.val

        good_subset = data.loc[(np.isclose(data['P_max'], pmax - 8.2))
                               & np.isclose(data['f1'], f1)
                               & np.isclose(data['f2'], f2)
                               & np.isclose(data['f3'], f3)]

        l = ax.scatter(good_subset['Pr_1'], good_subset['Pr_2'], good_subset['Pr_3'],
                       c=good_subset[RADIO_LABELS.get(radlabel)], norm=norm, cmap='jet')
        cb.update_normal(l)
        cb.set_label(radlabel)

        fig.canvas.draw_idle()

    spmax.on_changed(update)
    sf1.on_changed(update)
    sf2.on_changed(update)
    sf3.on_changed(update)

    def colorfunc(label):
        ax.clear()
        radlabel = label

        cmin = min(data[RADIO_LABELS.get(label)])
        cmax = max(data[RADIO_LABELS.get(label)])
        norm = get_norm(radlabel, cmin, cmax)

        ax.set_xlabel('Pr_1')
        ax.set_ylabel('Pr_2')
        ax.set_zlabel('Pr_3')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        pmax = spmax.val
        f1 = sf1.val
        f2 = sf2.val
        f3 = sf3.val

        good_subset = data.loc[(np.isclose(data['P_max'], pmax - 8.2))
                               & np.isclose(data['f1'], f1)
                               & np.isclose(data['f2'], f2)
                               & np.isclose(data['f3'], f3)]

        l = ax.scatter(good_subset['Pr_1'], good_subset['Pr_2'], good_subset['Pr_3'],
                       c=good_subset[RADIO_LABELS.get(radlabel)], norm=norm, cmap='jet')
        cb.update_normal(l)
        cb.set_label(radlabel)

        fig.canvas.draw_idle()

    radio.on_clicked(colorfunc)

    plt.show()


if __name__ == '__main__':
    df = pd.read_json('./gridsearch_data.json')
    plot_app(df)
