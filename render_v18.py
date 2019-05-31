# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This software is Copyright © 2019 The Regents of the University of California. All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation for educational, research and
# non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs appear in all copies. Permission
# to make commercial use of this software may be obtained by contacting:
#
# Office of Innovation and Commercialization
# 9500 Gilman Drive, Mail Code 0910
# University of California
# La Jolla, CA 92093-0910
# (858) 534-5815
# invent@ucsd.edu

# This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are supplied “as is”, without any accompanying services from The Regents.
# The Regents does not warrant that the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively
# on the program for any reason.

# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
# DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
import os
import shutil
import sys
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
Thermodynamic Neural Network Video, Image and Plot generator
'''

def display(folder_name, filename, show_video, save_state_video, save_change_video, save_images):
    '''
    Function to display and save videos and save image frames from simulation data in stored in 'filename'
    '''

#   Initialize variables
    print('Thermodynamic Neural Network Simulation Viewer')
    delay_between_frames = 50  # milliseconds (1 means as fast as possible)
    video_fps = 20

#   Set color values for image components
    logic_bgr_solved =   [int(255*x) for x in [0.0, 0.8, 0.0]]
    logic_bgr_unsolved = [int(255*x) for x in [0.1, 1.0, 1.0]]
    network_bgr_high =   [int(255*x) for x in [1.0, 1.0, 1.0]]
    network_bgr_low =    [int(255*x) for x in [0.0, 0.0, 0.0]]
    network_bgr_avg = [(network_bgr_high[j] + network_bgr_low[j])//2 for j in range(3)]
    network_bgr_dif = [(network_bgr_high[j] - network_bgr_low[j])//2 for j in range(3)]

#   Define node type lists
    network_node_list = ['binary', 'ternary', 'x-nary', 'continuous', 'ternary_sat', 'binary_sat', 'cortical']
    logic_node_list = ['or_sat', 'bias']

#   Read in the nodes from the simulation data file
    nodes=[]
    simulation = open(filename)
    line = ''
    while not line.startswith('Node Index'):
        line = simulation.readline().rstrip()
    line = simulation.readline().rstrip()
    while not line == '':
        (index, node_type, x, y) = line.split(', ')
        if node_type in network_node_list: node_class = 'network'
        if node_type in logic_node_list: node_class = 'logic'
        entry = {'node_class': node_class, 'node_type': node_type, 'x': x, 'y': y}
        nodes.append(entry)
        line = simulation.readline().rstrip()
    max_x = max_y = int(np.sqrt(len(nodes)))
    print('Successfully read in ' + str(len(nodes)) + ' nodes')

#   Pick node size in pixels for display, video and image files / = 40 to include text on nodes in video
    if 1 < max_y <= 25: node_width = node_height = 40
    if 25 < max_y <= 50: node_width = node_height = 20
    if 50 < max_y <= 100: node_width = node_height = 10
    if 100 < max_y <= 200: node_width = node_height = 5
    if 200 < max_y <= 500: node_width = node_height = 2
    if 500 < max_y <= 1000: node_width = node_height = 1
    resolution = (node_width * max_x, node_height * max_y)

#   Setup directories for saving video and image files
    image_dir = folder_name + '\\images'
    os.mkdir(image_dir)
    if save_state_video:
        state_video_filename = image_dir + '\\state_video.avi'
        print('Saving State Video to File: ' + '"' + state_video_filename + '"')
        state_video_recorder = cv2.VideoWriter()
        retval = state_video_recorder.open(state_video_filename, 1196444237, video_fps, resolution)
        assert(retval)
    if save_change_video:
        change_video_filename = image_dir + '\\change_video.avi'
        print('Saving Change Video to File: ' + '"' + change_video_filename + '"')
        change_video_recorder = cv2.VideoWriter()
        retval = change_video_recorder.open(change_video_filename, 1196444237, video_fps, resolution)
        assert(retval)

#   Read node states and create images
    frame = 0
    line = ''
    while not line.startswith('END'):
        while not line.startswith('node id,'):
            line = simulation.readline().rstrip()
        line = simulation.readline().rstrip()
        nodes_states = []
        nodes_state_changes = []
        nodes_solution = []
        while not line == '':
            (node_id, energy, state, state_change, entropy, solution, dissipation, transport) = line.split(', ')
            state = float(state)
            if abs(state)>1.0: state /= abs(state)
            nodes_states.append(state)
            state_change = float(state_change)
            if abs(state_change)>1.0: state_change /= abs(state_change)
            nodes_state_changes.append(state_change)
            nodes_solution.append(solution)
            line = simulation.readline().rstrip()
        line = simulation.readline().rstrip()
        frame += 1
        state_image = np.zeros((max_x, max_y, 3), dtype=np.uint8)
        change_image = np.zeros((max_x, max_y, 3), dtype=np.uint8)
        solved = True
        for i in range(len(nodes)):
            x = int(nodes[i]['x'])
            y = int(nodes[i]['y'])
            node_class = nodes[i]['node_class']
            if node_class == 'network':
                state_color = [network_bgr_avg[j] + network_bgr_dif[j] * nodes_states[i] for j in range(3)]
                change_color = [network_bgr_avg[j] + network_bgr_dif[j] * nodes_state_changes[i] for j in range(3)]
            if node_class == 'logic':
                if nodes_solution[i] == 'True':
                    state_color = change_color = logic_bgr_solved
                if nodes_solution[i] == 'False':
                    state_color = change_color = logic_bgr_unsolved
                    solved = False
            state_image[int(x), int(y)] = state_color
            change_image[int(x), int(y)] = change_color
        state_image = cv2.resize(state_image, resolution, interpolation=cv2.INTER_NEAREST)
        change_image = cv2.resize(change_image, resolution, interpolation=cv2.INTER_NEAREST)

#       Label nodes within the images
        if node_width == 40:
            for i in range(len(nodes)):
                x = int(nodes[i]['y'])
                y = int(nodes[i]['x'])
                if nodes[i]['node_type'] == 'or_sat':
                    cv2.putText(img=state_image, text='O', org=(x*node_width+12, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,0), thickness=1)
                    cv2.putText(img=change_image, text='O', org=(x*node_width+12, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,0), thickness=1)
                if nodes[i]['node_type'] == 'bias':
                    if nodes_states[i] > 0.0:
                        cv2.putText(img=state_image, text='+', org=(x*node_width+8, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.90, color=(0,0,0), thickness=2)
                        cv2.putText(img=change_image, text='+', org=(x*node_width+8, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.90, color=(0,0,0), thickness=2)
                    else:
                        cv2.putText(img=state_image, text='-', org=(x*node_width+8, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.90, color=(0,0,0), thickness=2)
                        cv2.putText(img=change_image, text='-', org=(x*node_width+8, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.90, color=(0,0,0), thickness=2)
                if nodes[i]['node_type'] == 'ternary_sat':
                    cv2.putText(img=state_image, text='S', org=(x*node_width+13, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,255,255), thickness=1)
                    cv2.putText(img=change_image, text='S', org=(x*node_width+13, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,255,255), thickness=1)
                if nodes[i]['node_type'] == 'binary_sat':
                    if nodes_states[i] < 0.0:
                        cv2.putText(img=state_image, text='S', org=(x*node_width+13, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,255,255), thickness=1)
                        cv2.putText(img=change_image, text='S', org=(x*node_width+13, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(255,255,255), thickness=1)
                    else:
                        cv2.putText(img=state_image, text='S', org=(x*node_width+13, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,0), thickness=1)
                        cv2.putText(img=change_image, text='S', org=(x*node_width+13, y*node_height+27), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0,0,0), thickness=1)
        if node_width == 20:
            for i in range(len(nodes)):
                x = int(nodes[i]['y'])
                y = int(nodes[i]['x'])
                if nodes[i]['node_type'] == 'or_sat':
                    cv2.putText(img=state_image, text='O', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,0), thickness=1)
                    cv2.putText(img=change_image, text='O', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,0), thickness=1)
                if nodes[i]['node_type'] == 'bias':
                    if nodes_states[i] > 0.0:
                        cv2.putText(img=state_image, text='+', org=(x*node_width+4, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0,0,0), thickness=1)
                        cv2.putText(img=change_image, text='+', org=(x*node_width+4, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0,0,0), thickness=1)
                    else:
                        cv2.putText(img=state_image, text='-', org=(x*node_width+4, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0,0,0), thickness=1)
                        cv2.putText(img=change_image, text='-', org=(x*node_width+4, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45, color=(0,0,0), thickness=1)
                if nodes[i]['node_type'] == 'ternary_sat':
                    cv2.putText(img=state_image, text='S', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255,255,255), thickness=1)
                    cv2.putText(img=change_image, text='S', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255,255,255), thickness=1)
                if nodes[i]['node_type'] == 'binary_sat':
                    if nodes_states[i] < 0.0:
                        cv2.putText(img=state_image, text='S', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255,255,255), thickness=1)
                        cv2.putText(img=change_image, text='S', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255,255,255), thickness=1)
                    else:
                        cv2.putText(img=state_image, text='S', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,0), thickness=1)
                        cv2.putText(img=change_image, text='S', org=(x*node_width+6, y*node_height+13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,0,0), thickness=1)

#       Display images, store video, store frame images
        if show_video:
            cv2.imshow(folder_name, state_image)
            cv2.waitKey(delay_between_frames)
        if save_state_video: state_video_recorder.write(state_image)
        if save_change_video: change_video_recorder.write(change_image)
        if save_images:
            if frame%save_images == 0 or not solved:
                state_image_filename = image_dir + '\\state_image_' + str(frame) + '.png'
                cv2.imwrite(state_image_filename, state_image)

#   Close files and windows
    if save_state_video: state_video_recorder.release()
    if save_change_video: change_video_recorder.release()
    simulation.close()
    cv2.destroyAllWindows()


def makeplots(folder_name, filename):
    '''
    Function to save plots from simulation statistics in stored in 'filename'
    '''

#   Initialize variables
    print('\nThermodynamic Neural Network Plot Saver\n')
    plot_data = open(filename)
    data_type_list = ['energy', 'synapse2', 'state_change', 'solved', 'entropy', 'dissipation', 'transport', 'order', 'quality']
    data_dict = {item: [] for item in data_type_list}
    time_list = []

#   Read in Data
    line = ''
    while not line.startswith('Time,'):
        line = plot_data.readline().rstrip()
    line = plot_data.readline().rstrip()
    while not line == 'END':
        (time, energy, synapse2, state_change, solved, entropy, dissipation, transport, quality, order) = line.split(', ')
        time_list.append(float(time))
        data_dict['energy'].append(float(energy))
        data_dict['synapse2'].append(float(synapse2))
        data_dict['state_change'].append(float(state_change))
        data_dict['solved'].append(float(solved))
        data_dict['entropy'].append(float(entropy))
        data_dict['dissipation'].append(float(dissipation))
        data_dict['transport'].append(float(transport))
        data_dict['order'].append(float(order))
        data_dict['quality'].append(float(quality))
        line = plot_data.readline().rstrip()
    print('Successfully read in ' + str(len(time_list)) + ' simulation steps\n')

#   Setup directory
    plot_dir = folder_name + '\\plots'
    os.mkdir(plot_dir)

#   Save plots
    for item in data_type_list:
        filename = plot_dir + '\\' + item + '.png'
        fig, ax = plt.subplots()
        ax.plot(time_list, data_dict[item], 'b--')
        ax.set(xlabel='Simulation Step', ylabel=item, title = item + ' vs time')
        ax.grid()
        fig.savefig(filename)
        print('"' + filename + '"' + ' successfully saved')

#   save energy / dissipation plot
    filename = plot_dir + '\\energy_dissipation.png'
    fig, ax = plt.subplots()
    ax.plot(time_list, data_dict['energy'], 'b--', label='Energy')
    ax.plot(time_list, data_dict['dissipation'], 'g--', label='Dissipation')
    ax.set(xlabel='Simulation Step', ylabel='Energy', title='Average Node Energy and Dissipation vs Time')
    ax.grid()
    ax.legend()
    fig.savefig(filename)
    print('"' + filename + '"' + ' successfully saved')

#   save energy / dissipation / transport plot
    filename = plot_dir + '\\energy_dissipation_transport.png'
    fig, ax = plt.subplots()
    ax.plot(time_list, data_dict['energy'], 'b--', label='Energy')
    ax.plot(time_list, data_dict['dissipation'], 'g--', label='Dissipation')
    ax.plot(time_list, data_dict['transport'], 'r--', label='Transport')
    ax.set(xlabel='Simulation Step', ylabel='Energy', title='Average Node Energy, Dissipation and Transport vs Time')
    ax.grid()
    ax.legend()
    fig.savefig(filename)
    print('"' + filename + '"' + ' successfully saved')

#   close files
    plot_data.close()


if __name__ == '__main__':

    show_video = input('\nView Video (y / N)?') == 'y'                                              # if True renders the network video to the display
    save_state_video = input('\nSave Network State Video (y / N)?') == 'y'                          # if True saves the network video to a file
    save_change_video = input('\nSave Network Change Video (y / N)?') == 'y'                        # if True saves the network video to a file
    save_images = input('\nSave every Nth frame (input N)?')                                        # Saves every Nth image of the state video to a file - 0 saves none
    display('', 'network_data.txt', show_video, save_state_video, save_change_video, save_images)
    if input('\Make Plots (y / N)?') == 'y': makeplots('', 'plot_data.txt')
