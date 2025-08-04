import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, voronoi_plot_2d   


def animate(anim_params):
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['text.usetex'] = True
    ref_state_list = anim_params['ref_state_list']
    agents_init_state = anim_params['agents_init_state']
    agents_state_list = anim_params['agents_state_list']
    agents_control_list = anim_params['agents_control_list'] 
    obstacles = anim_params['obstacles']
    num_frames = anim_params['num_frames']
    max_iter = anim_params['max_iter']
    pred_horizon = anim_params['pred_horizon'] 
    vor = anim_params['vor']
    vcentroids = anim_params['vcentroids']
    save = anim_params['save'] 
    adj_matrix = anim_params['adj_matrix']
    

    def create_triangle(state=[0,0,0], h=0.2, w=0.15, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])
        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]


    # Function to create a gradient-filled circle
    def radial_gradient_circle(ax, center_x, center_y, radius, colormap='viridis'):
        """
        Creates a radial gradient circle.
        """
        # Create a meshgrid for the circle
        x, y = np.meshgrid(np.linspace(center_x - radius, center_x + radius, 100),
                        np.linspace(center_y - radius, center_y + radius, 100))
        # Calculate the distance from the center for each point
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Normalize the distance to be between 0 and 1
        r = np.clip(r, 0, radius) / radius
        # Create a colormap
        cmap = plt.get_cmap(colormap).reversed()
        # Map the distance to the colormap
        colors = cmap(r)
        # Plot the circle
        ax.imshow(colors, extent=[center_x - radius, center_x + radius, center_y - radius, center_y + radius], alpha=0.1)
        # Set aspect to 'equal' to ensure the circle looks circular
        ax.set_aspect('equal')



    def init():
        return path_list, horizon_list
    
    def animate(i):
        ax.clear()
        for k in range(n_agents):
            # get variables
            x = agents_state_list[k][0, 0, i]
            y = agents_state_list[k][1, 0, i]
            th = agents_state_list[k][2, 0, i]

            # update path
            if i == 0:
                path_list[k].set_data(np.array([]), np.array([]))
            x_new = np.hstack((path_list[k].get_xdata(), x))
            y_new = np.hstack((path_list[k].get_ydata(), y))
            path, = ax.plot(x_new, y_new, 'r', linewidth=2)
           
            # update horizon
            x_new = agents_state_list[k][0, :, i]
            y_new = agents_state_list[k][1, :, i]
            horizon, = ax.plot(x_new, y_new, 'x-g', alpha=0.5)
            
            #current_state_list[k].set_xy(create_triangle([x, y, th], update=True))
            current_triangle = create_triangle([x, y, th])
            current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='b')
            current_state = current_state[0]

            # Draw a transparent circle
            radial_gradient_circle(ax, x, y, radius=1.5, colormap='Reds')

        if (vor != None):
            den = num_frames/max_iter
            # Show Centroids
            for vcentroid in vcentroids[int(i/den)]:
                ax.plot(vcentroid[0], vcentroid[1], 'o', color='blue')

            voronoi_plot_2d(vor[int(i/den)], ax=ax, show_vertices=False, show_points = False, line_colors='orange', line_width=2)
            for j in range(adj_matrix.shape[0]):
                for k in range(adj_matrix.shape[1]):
                    if (adj_matrix[j][k]==1):
                        centroid_1 = vcentroids[int(i/den)][j]
                        centroid_2 = vcentroids[int(i/den)][k]
                        x = [centroid_1[0], centroid_2[0]]
                        y = [centroid_1[1], centroid_2[1]]
                        ax.plot(x, y, '--r', alpha=0.1)
                
            legend_elements = [ Line2D([0], [0], marker='>', color='b', markerfacecolor='b', markersize=15, label='Robots'),
                                Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',),
                                Line2D([0], [0], linestyle='-',color='orange', markerfacecolor='r', markersize=15,label='Voronoi Partition',),
                                Line2D([0], [0], linestyle='--',color='r', markerfacecolor='r', markersize=15, alpha = 0.2, label='Desired Bearing',),
                                Line2D([0], [0], linestyle='--',color='k', markerfacecolor='k', markersize=15,alpha = 0.2, label='Current Bearing',),
                            ]

            ax.legend(handles=legend_elements, loc='upper right', fontsize = 10)            
        else:
            legend_elements = [ Line2D([0], [0], marker='>', color='b', markerfacecolor='b', markersize=15, label='Robots'),
                                Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',)
                            ]

            ax.legend(handles=legend_elements, loc='upper right', fontsize = 10)   

        ax.set_xlim(0,5)
        ax.set_ylim(0,5)
        ax.set_xlabel('x position', fontsize =12)
        ax.set_ylabel('y position', fontsize =12)       
        for j in range(adj_matrix.shape[0]):
            for k in range(adj_matrix.shape[1]):
                if (adj_matrix[j][k]==1):
                    state_agent_1 = agents_state_list[j][:, 0, i]
                    state_agent_2 = agents_state_list[k][:, 0, i]
                    x = [state_agent_1[0], state_agent_2[0]]
                    y = [state_agent_1[1], state_agent_2[1]]
                    ax.plot(x, y, '--k', alpha=0.1)

        plt.tight_layout()
        return path, horizon

    # create figure and axes
    n_agents = agents_init_state.shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))
    # create lines:
    #   path
    path_list = []
    ref_path_list = []
    horizon_list = []
    current_state_list = []
  
    for k in range(n_agents):
        path, = ax.plot([], [], 'r', linewidth=2)
        ref_path, = ax.plot([], [], 'b', linewidth=2)
        horizon, = ax.plot([], [], 'x-g', alpha=0.5)
        current_triangle = create_triangle(agents_init_state[k, :])
        current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='y')
        current_state = current_state[0]

        path_list.append(path)
        ref_path_list.append(ref_path)
        horizon_list.append(horizon)
        current_state_list.append(current_state)


    
    red_cmp = plt.get_cmap('Reds', 256)
    red_cmp = ListedColormap(red_cmp(np.linspace(0, 0.3, 256)))
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=red_cmp),
             ax=ax, orientation='vertical',fraction=0.046, pad=0.04, label='Percent Coverage')
    
    sim = animation.FuncAnimation(
        fig=fig,
        func = animate,
        init_func=init,
        frames=num_frames,
        interval=100,
        blit=False,
        repeat=False
    )
    if save == True:
        sim.save(f'{anim_params['file_name']}.mp4', writer='ffmpeg', fps=10)

    plt.show()    
    return sim

def plot_figure(anim_params):   
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['text.usetex'] = True

    ref_state_list = anim_params['ref_state_list']
    agents_init_state = anim_params['agents_init_state']
    agents_state_list = anim_params['agents_state_list']
    agents_control_list = anim_params['agents_control_list'] 
    obstacles = anim_params['obstacles']
    num_frames = anim_params['num_frames']
    max_iter = anim_params['max_iter']
    pred_horizon = anim_params['pred_horizon'] 
    vor = anim_params['vor']
    vcentroids = anim_params['vcentroids']
    save = anim_params['save'] 
    adj_matrix = anim_params['adj_matrix']

    def create_triangle(state=[0,0,0], h=0.2, w=0.15, update=False):
            x, y, th = state
            triangle = np.array([
                [h, 0   ],
                [0,  w/2],
                [0, -w/2],
                [h, 0   ]
            ]).T
            rotation_matrix = np.array([
                [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)]
            ])
            coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
            if update == True:
                return coords
            else:
                return coords[:3, :]


    # Function to create a gradient-filled circle
    def radial_gradient_circle(ax, center_x, center_y, radius, colormap='viridis'):
        """
        Creates a radial gradient circle.
        """
        # Create a meshgrid for the circle
        x, y = np.meshgrid(np.linspace(center_x - radius, center_x + radius, 100),
                        np.linspace(center_y - radius, center_y + radius, 100))
        # Calculate the distance from the center for each point
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Normalize the distance to be between 0 and 1
        r = np.clip(r, 0, radius) / radius
        # Create a colormap
        cmap = plt.get_cmap(colormap).reversed()
        # Map the distance to the colormap
        colors = cmap(r)
        # Plot the circle
        ax.imshow(colors, extent=[center_x - radius, center_x + radius, center_y - radius, center_y + radius], alpha=0.1)
        # Set aspect to 'equal' to ensure the circle looks circular
        ax.set_aspect('equal')



    # create figure and axes
    n_agents = 4
    fig, ax = plt.subplots(1,3, figsize=(12, 6))
    # create lines:
    #   path
    path_list = []
    ref_path_list = []
    horizon_list = []
    current_state_list = []

    indexes = [0, 50, 149]
    for i, idx in enumerate(indexes):
        for k in range(n_agents):
            # get variables
            x = agents_state_list[k][0, 0, idx]
            y = agents_state_list[k][1, 0, idx]
            th = agents_state_list[k][2, 0, idx]
            
            # update horizon
            x_new = agents_state_list[k][0, :, idx]
            y_new = agents_state_list[k][1, :, idx]
            horizon, = ax[i].plot(x_new, y_new, 'x-g', alpha=0.5)
            
            current_triangle = create_triangle([x, y, th])
            current_state = ax[i].fill(current_triangle[:, 0], current_triangle[:, 1], color='b')
            current_state = current_state[0]

            # Draw a transparent circle
            radial_gradient_circle(ax[i], x, y, radius=1.5, colormap='Reds')

    
        den = num_frames/max_iter
        # Show Centroids
        for vcentroid in vcentroids[int(idx/den)]:
            ax[i].plot(vcentroid[0], vcentroid[1], 'o', color='blue')

        voronoi_plot_2d(vor[int(idx/den)], ax=ax[i], show_vertices=False, show_points = False, line_colors='orange', line_width=2)
        for j in range(adj_matrix.shape[0]):
            for k in range(adj_matrix.shape[1]):
                if (adj_matrix[j][k]==1):
                    centroid_1 = vcentroids[int(idx/den)][j]
                    centroid_2 = vcentroids[int(idx/den)][k]
                    x = [centroid_1[0], centroid_2[0]]
                    y = [centroid_1[1], centroid_2[1]]
                    ax[i].plot(x, y, '--r', alpha=0.1)
            
                

        ax[i].set_xlim(0,5)
        ax[i].set_ylim(0,5)
        ax[i].set_xlabel('x position [m]', fontsize =12)
        ax[i].set_ylabel('y position [m]', fontsize =12)
        ax[i].set_rasterized(True)
        
        for j in range(adj_matrix.shape[0]):
            for k in range(adj_matrix.shape[1]):
                if (adj_matrix[j][k]==1):
                    state_agent_1 = agents_state_list[j][:, 0, idx]
                    state_agent_2 = agents_state_list[k][:, 0, idx]
                    x = [state_agent_1[0], state_agent_2[0]]
                    y = [state_agent_1[1], state_agent_2[1]]
                    ax[i].plot(x, y, '--k', alpha=0.1)


    legend_elements = [ Line2D([0], [0], marker='>', color='b', markerfacecolor='b', markersize=15, label='Robots'),
                        Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',),
                        Line2D([0], [0], linestyle='-',color='orange', markerfacecolor='r', markersize=15,label='Voronoi Partition',),
                        Line2D([0], [0], linestyle='--',color='r', markerfacecolor='r', markersize=15, alpha = 0.2, label='Desired Bearing',),
                        Line2D([0], [0], linestyle='--',color='k', markerfacecolor='k', markersize=15,alpha = 0.2, label='Current Bearing',),
                    ]
    # plt.tight_layout()
    fig.legend(handles=legend_elements, loc='upper right',bbox_to_anchor=[0.94, 0.82],  fontsize = 13, ncol=5)  
    red_cmp = plt.get_cmap('Reds', 256)
    red_cmp = ListedColormap(red_cmp(np.linspace(0, 0.3, 256)))

    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=red_cmp),
                ax=ax, orientation='vertical',fraction=0.015, pad=0.04, label='Percent Coverage')
    if save == True:
        plt.savefig(anim_params['file_name'])
    plt.show()     
    

