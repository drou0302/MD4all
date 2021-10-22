from __future__ import print_function
from contextlib import redirect_stdout, redirect_stderr
from os import path
from sys import platform
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from scipy.spatial.distance import squareform

import PySimpleGUI as sg
import sys
import os
import traceback
import webbrowser
import json
import subprocess
import time
import mdtraj
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy

matplotlib.use('Agg')

def main():
    sg.theme('System Default 1')
    platforms = ['Reference', 'CPU', 'CUDA', 'OpenCL']

    layout = [[sg.Text('Welcome to MD4all', font=('Helvetica', 20, 'bold'), justification = 'center', size=(45,1))],
              [sg.Text('Choose your input file type and indicate the location:', font=('Helvetica', 10, 'bold'))],
              [sg.Checkbox('PDB File', size=(15, 1)),
               sg.Text('Location of your pdb file: ', size=(25,1)), sg.Input(size=(45,1))],
              [sg.Checkbox('Amber Files', size=(15, 1)), sg.Text('Location of coordinate file (inpcrd)', size=(25,1)),
              sg.Input(size=(45,1))],
              [sg.Text('', size=(18, 1)), sg.Text('Location of topology file (prmtop):', size=(25,1)), sg.Input(size=(45,1))],
              [sg.Text('Choose the platform to run the simulation:', font=('Helvetica', 10, 'bold'), size = (45,1)),
               sg.Combo(platforms, default_value='Reference', size=(30,30))],
              [sg.Text('Do you want to perform a minimization first? ', font=('Helvetica', 10, 'bold'), size=(45,1)),
               sg.Checkbox('Yes', default=True), sg.Checkbox('No')],
              [sg.Text('If yes, please indicate the parameters for the minimization:', size=(45, 1))],
              [sg.Text('Maximum iteration steps:', size=(45,1)), sg.Input(default_text='-')],
              [sg.Text('Energy tolerance for convergence:', size=(45, 1)), sg.Input(default_text='10')],
              [sg.Text('Save minimized state?', font=('Helvetica', 10), size=(45,1)),
               sg.Checkbox('Yes', default=True), sg.Checkbox('No')],
              [sg.Text('Choose the parameters for the simulation: ', font=('Helvetica', 10, 'bold'))],
              [sg.Text('Temperature:', size=(45,1)), sg.Input(default_text='300')],
              [sg.Text('Time (ns):', size=(45,1)), sg.Input(default_text='2')],
              [sg.Text('Write results every (ns):', size=(45,1)), sg.Input(default_text='0.2')],
              [sg.Text('Do you want to analyse the results directly?', font=('Helvetica', 10, 'bold'), size=(45,1)),
               sg.Checkbox('Yes'), sg.Checkbox('No')],
              [sg.Text('OUTPUT:', font=('Helvetica', 10, 'bold'))],
              [sg.Output(key='-output-', size=(100, 10))],
              [sg.Button('Run'), sg.Button('Exit')]]
    
    window_main = sg.Window('MD4all', layout)

    while True:
        event, values = window_main.read()
        if event in (None, 'Exit'):
            break
        elif event == 'Run':
            checks = []
            window_main['-output-'].Update('')
            if values[0] and values[0] != values[2]:
                input_format = 'PDB'
                if len(values[1]) == 0:
                    checks.append('Indicate the pdb file, please!')
                else:
                    pdbfile = values[1]
                    coordinate_file = ''
                    topology_file = ''
            elif values[2] and values[0] != values[2]:
                input_format = 'AMBER'
                if len(values[3]) == 0:
                    checks.append('Indicate the coordinate file, please!')
                if len(values[4]) == 0:
                    checks.append('Indicate the parameter file, please!')
                else:
                    coordinate_file = values[3]
                    topology_file = values[4]
                    pdbfile=''
            else:
                checks.append('Indicate either pdb or amber files.')


            platform_sim = values[5]

            if values[6] != values[7]:
                if values[6]:
                    min_sim = 'Yes'
                    min_steps = values[8]
                    max_tolerance = values[9]
                    if values[10] != values[11]:
                        if values[10]:
                            save_min = 'YES'
                        else:
                            save_min = 'NO'
                else:
                    min_sim = 'No'

            if values[12] == '':
                checks.append('Indicate the temperature (K)')
            else:
                temp_sim = float(values[12])

            if values[13] == '':
                checks.append('Indicate the time of simulation')
            else:
                time_sim = float(values[13]) * 500000

            if values[14] == '':
                checks.append('Indicate how often do you want to write the results')
            else:
                write_sim = float(values[14]) * 500000
            if values[15] != values[16]:
                if values[15]:
                    analysis_answer = 'YES'
                else:
                    analysis_answer = 'NO'
            else:
                checks.append('Indicate if you want to analyse the results or not!')
            if len(checks) == 0:
                try:
                   md(input_format, pdbfile, coordinate_file, topology_file, platform_sim, min_sim,
                   temp_sim, time_sim, write_sim, min_steps, max_tolerance, save_min)
                   if analysis_answer == 'YES':
                       analysis_md(input_format, pdbfile, topology_file)
                except BaseException as error:
                    print(error)
                    window_main.refresh()
            else:
                for errors in checks:
                    print(errors)
                    redirect_stdout(window_main['-output-'])
                    window_main.refresh()

def md(input_format, pdbfile, coordinate_file, topology_file, platform_sim, min_sim, temp_sim, time_sim, write_sim,
       min_steps, max_tolerance, save_min):
    print('Loading files...')
    if input_format == 'PDB':
        pdb = PDBFile(pdbfile)
    elif input_format == 'AMBER':
        prmtop = AmberPrmtopFile(topology_file)
        inpcrd = AmberInpcrdFile(coordinate_file)
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    print('Creating the system...')
    if input_format == 'PDB':
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1*nanometer,
                                     rigidWater=True, ewaldErrorTolerance=0.0005)
        temp = int(temp_sim)
        integrator = LangevinIntegrator(temp * kelvin, 1 / picosecond, 0.002 * picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
    else:
        system = prmtop.createSystem(nonbondedMethod=NoCutoff, nonbondedCutoff=1 * nanometer,
                                         rigidWater=True, ewaldErrorTolerance=0.0005)
        temp = int(temp_sim)
        integrator = LangevinIntegrator(temp * kelvin, 1 / picosecond, 0.002 * picoseconds)
        simulation = Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)
        if inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    platform = Platform.getPlatformByName(platform_sim)


    if min_sim == 'Yes':
        print('Starting minimization...')
        if min_steps == '-':
            simulation.minimizeEnergy(tolerance=int(max_tolerance) * kilojoule / mole)
        elif min_steps != '-':
            simulation.minimizeEnergy(tolerance=10 * kilojoule / mole, maxIterations=int(min_steps))
        else:
            simulation.minimizeEnergy()
        positions = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, positions, open('minimized.pdb', 'w'))
        print('Finished minimization!')

    print('Starting equilibration of the system...')
    mdsteps = int(time_sim)
    repsteps = int(write_sim)
    simulation.reporters.append(DCDReporter('output.dcd', repsteps))
    simulation.reporters.append(StateDataReporter('ouptut.log', repsteps, step=True, time=True, potentialEnergy=True,
                                                  kineticEnergy=True,totalEnergy=True,temperature=True, progress=True,
                                                  volume=True,density=True, remainingTime=True, speed=True,
                                                  totalSteps=mdsteps, separator='\t'))
    simulation.reporters.append(StateDataReporter(stdout, 50000, step=True, potentialEnergy=True, temperature=True,
                                                  progress=True, remainingTime=True, speed=True, totalSteps=mdsteps,
                                                  separator='\t'))
    simulation.step(mdsteps)
    print('MD finished!')

def analysis_md(input_format, pdbfile, prmtop):
    if input_format == 'PDB':
        top_from_pdb = mdtraj.load(pdbfile).topology
        trajectory = mdtraj.load('output.dcd', top=top_from_pdb)
    else:
        trajectory = mdtraj.load('output.dcd', top=prmtop)

    traj = trajectory.remove_solvent()
    atom_indices = []
    for a in traj.topology.atoms:
        if a.element.symbol != 'H':
            atom_indices.append(a.index)

    distances = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        distances[i] = mdtraj.rmsd(traj, traj, i, atom_indices=atom_indices)
    beta = 1
    index = np.exp(-beta * distances / distances.std()).sum(axis=1).argmax()
    centroid = traj[index]
    centroid.save_pdb('centroid.pdb')
    rmsds_to_centroid = mdtraj.rmsd(traj, traj, index)
    heavy_atoms = [atom.index for atom in traj.topology.atoms if atom.element.symbol != 'H']
    heavy_rmds_to_min = mdtraj.rmsd(traj, traj, index, atom_indices=heavy_atoms)

    plt.figure();
    x2 = 0.1 * traj.time
    plt.plot(x2, heavy_rmds_to_min);
    plt.legend();
    plt.title('RMSD');
    plt.xlabel('Simulation time (ns)');
    plt.ylabel('RMSD (nm)');
    plt.savefig('rmsd_min.png');

    assert np.all(distances - distances.T < 1e-3)
    reduced_distances = squareform(distances, checks=False)
    linkage = scipy.cluster.hierarchy.linkage(reduced_distances, method='average')

    plt.figure();
    plt.title('RMSD Average linkage hierarchical clustering');
    plt.ylabel('RMSD (nm)');
    plt.xlabel('Clusters');
    _ = scipy.cluster.hierarchy.dendrogram(linkage, truncate_mode='lastp', no_labels=True, count_sort='descendent');
    plt.savefig('clusters.png');

    print('Analysis finished.')

if __name__ == "__main__":
    main()
