import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

class MicrobiomeGBAModel:
    """
    Extended Gut-Brain Axis model incorporating microbiome community dynamics via
    Lotka-Volterra equations, stochastic fluctuations, and realistic homeostatic
    mechanisms.
    
    This model captures:
    1. Competition and cooperation between bacterial taxa
    2. Environmental perturbations (diet, antibiotics, stress)
    3. Host-microbiome feedbacks via metabolites and immune signals
    4. Critical transitions in both microbial ecology and host metabolism
    """
    
    def __init__(self, params=None):
        """
        Initialize model with default or custom parameters
        
        Args:
            params: Optional dictionary of parameters to override defaults
        """
        # Default parameters with units
        self.params = {
            # Host parameters
            'k_secr_glp1': 0.2,      # GLP-1 secretion rate [concentration/hour]
            'k_max_glp1': 1.0,       # GLP-1 saturation [concentration]
            'k_deg_glp1': 0.1,       # GLP-1 degradation rate [1/hour]
            'k_vagus_glp1': 0.15,    # Vagus-mediated GLP-1 stimulation [1/hour]
            
            'k_ghrelin': 0.3,        # Baseline ghrelin secretion [concentration/hour]
            'k_max_ghrelin': 1.0,    # Ghrelin saturation [concentration]
            'k_sat_ghrelin': 0.2,    # Suppression by satiety hormones [1/concentration·hour]
            
            # Neural parameters
            'k_dist': 0.2,           # Gut distension sensitivity [1/hour]
            'k_deg_vagus': 0.1,      # Vagal signal decay [1/hour]
            'k_stress': 0.3,         # Stress modulation of vagus [1/hour]
            
            # Energy balance parameters
            'k_energy_intake': 0.3,  # Energy absorption coefficient [energy/hour]
            'k_energy_expend': 0.2,  # Energy expenditure coefficient [1/hour]
            
            # SCFA parameters
            'k_deg_scfa': 0.15,      # SCFA degradation rate [1/hour]
            'k_scfa_butyrate': 0.8,  # Butyrate contribution to total SCFA effect [ratio]
            'k_scfa_propionate': 0.5, # Propionate contribution to total SCFA effect [ratio]
            'k_scfa_acetate': 0.3,   # Acetate contribution to total SCFA effect [ratio]
            
            # Microbiome parameters (Lotka-Volterra)
            # Base growth rates [1/hour]
            'r_bacteroides': 0.12,   # Bacteroides growth rate 
            'r_firmicutes': 0.15,    # Firmicutes growth rate
            'r_bifidobacteria': 0.08, # Bifidobacteria growth rate
            'r_proteobacteria': 0.20, # Proteobacteria growth rate
            
            # Carrying capacities [abundance]
            'K_bacteroides': 1.0,    # Bacteroides carrying capacity
            'K_firmicutes': 1.0,     # Firmicutes carrying capacity 
            'K_bifidobacteria': 0.6, # Bifidobacteria carrying capacity
            'K_proteobacteria': 0.4, # Proteobacteria carrying capacity
            
            # Interaction matrix [1/abundance·hour]
            # Format: effect of column species on row species
            # Negative: competition, Positive: facilitation
            'alpha_BB': -1.0,  # Bacteroides on Bacteroides (self-limitation)
            'alpha_BF': -0.4,  # Firmicutes on Bacteroides
            'alpha_BBi': 0.1,  # Bifidobacteria on Bacteroides
            'alpha_BP': -0.6,  # Proteobacteria on Bacteroides
            
            'alpha_FB': -0.4,  # Bacteroides on Firmicutes
            'alpha_FF': -1.0,  # Firmicutes on Firmicutes (self-limitation)
            'alpha_FBi': -0.2, # Bifidobacteria on Firmicutes
            'alpha_FP': -0.7,  # Proteobacteria on Firmicutes
            
            'alpha_BiB': 0.2,  # Bacteroides on Bifidobacteria
            'alpha_BiF': 0.3,  # Firmicutes on Bifidobacteria
            'alpha_BiBi': -1.0, # Bifidobacteria on Bifidobacteria (self-limitation)
            'alpha_BiP': -0.8,  # Proteobacteria on Bifidobacteria
            
            'alpha_PB': -0.2,  # Bacteroides on Proteobacteria
            'alpha_PF': -0.3,  # Firmicutes on Proteobacteria
            'alpha_PBi': -0.6, # Bifidobacteria on Proteobacteria
            'alpha_PP': -1.0,  # Proteobacteria on Proteobacteria (self-limitation)
            
            # Metabolite production rates [metabolite/abundance·hour]
            'k_butyrate_F': 0.35,    # Butyrate production by Firmicutes
            'k_butyrate_Bi': 0.15,   # Butyrate production by Bifidobacteria
            'k_propionate_B': 0.30,  # Propionate production by Bacteroides
            'k_acetate_B': 0.25,     # Acetate production by Bacteroides
            'k_acetate_F': 0.20,     # Acetate production by Firmicutes
            'k_acetate_Bi': 0.40,    # Acetate production by Bifidobacteria
            
            # Diet influence on bacterial growth [dimensionless]
            'k_fiber_B': 0.5,        # Fiber effect on Bacteroides
            'k_fiber_F': 0.7,        # Fiber effect on Firmicutes
            'k_fiber_Bi': 0.9,       # Fiber effect on Bifidobacteria
            'k_fiber_P': 0.1,        # Fiber effect on Proteobacteria
            'k_protein_B': 0.3,      # Protein effect on Bacteroides
            'k_protein_F': 0.4,      # Protein effect on Firmicutes
            'k_protein_P': 0.8,      # Protein effect on Proteobacteria
            'k_fat_F': 0.6,          # Fat effect on Firmicutes
            'k_fat_P': 0.5,          # Fat effect on Proteobacteria
            
            # External inputs [0-1 scale]
            'fiber_intake': 0.5,     # Dietary fiber
            'protein_intake': 0.6,   # Dietary protein
            'fat_intake': 0.4,       # Dietary fat
            'stress_level': 0.2,     # Cortisol/stress
            
            # Antibiotic effect [1/hour]
            'k_abx_B': 0.5,          # Antibiotic effect on Bacteroides
            'k_abx_F': 0.6,          # Antibiotic effect on Firmicutes
            'k_abx_Bi': 0.8,         # Antibiotic effect on Bifidobacteria
            'k_abx_P': 0.3,          # Antibiotic effect on Proteobacteria
            'abx_level': 0.0,        # Antibiotic concentration [0-1]
            
            # Noise parameters
            'sigma_glp1': 0.03,      # GLP-1 noise intensity
            'sigma_ghrelin': 0.05,   # Ghrelin noise intensity
            'sigma_vagus': 0.04,     # Vagus noise intensity
            'sigma_bacteria': 0.06,  # Bacterial abundance noise intensity
            'sigma_metabolites': 0.04, # Metabolite noise intensity
            'sigma_energy': 0.02,    # Energy noise intensity
            
            # Bistability parameters
            'bistable_threshold': 0.4, # Threshold for bistable behavior
            'feedback_strength': 1.8   # Strength of nonlinear feedback
        }
        
        # Update with custom parameters if provided
        if params is not None:
            self.params.update(params)
        
        # Initialize interaction matrix for Lotka-Volterra equations
        self._init_interaction_matrix()
    
    def _init_interaction_matrix(self):
        """Initialize the interaction matrix for the Lotka-Volterra model"""
        self.alpha_matrix = np.array([
            [self.params['alpha_BB'], self.params['alpha_BF'], self.params['alpha_BBi'], self.params['alpha_BP']],
            [self.params['alpha_FB'], self.params['alpha_FF'], self.params['alpha_FBi'], self.params['alpha_FP']],
            [self.params['alpha_BiB'], self.params['alpha_BiF'], self.params['alpha_BiBi'], self.params['alpha_BiP']],
            [self.params['alpha_PB'], self.params['alpha_PF'], self.params['alpha_PBi'], self.params['alpha_PP']]
        ])
    
    def drift_terms(self, t, y):
        """
        Calculate deterministic drift terms for the SDEs
        
        Args:
            t: time
            y: state vector [GLP1, Ghrelin, Vagus, Bacteroides, Firmicutes, Bifidobacteria, 
                            Proteobacteria, Butyrate, Propionate, Acetate, Energy]
        
        Returns:
            Array of drift terms for each state variable
        """
        # Extract state variables
        GLP1, Ghrelin, Vagus = y[0:3]
        Bacteroides, Firmicutes, Bifidobacteria, Proteobacteria = y[3:7]
        Butyrate, Propionate, Acetate = y[7:10]
        Energy = y[10]
        
        # Clamp bacterial abundances to non-negative values for calculations
        bacteria = np.maximum(y[3:7], 0)
        
        # Extract parameters for readability
        p = self.params
        
        # Bistability factor for critical transitions
        # This is vectorized as suggested in the feedback
        def bistable_factor(x, threshold, strength):
            sign = np.where(x > threshold, 1, -1)
            return 1.0 + strength * (x - threshold)**2 * sign
        
        # Calculate total SCFA effect (weighted sum of individual SCFAs)
        total_scfa_effect = (p['k_scfa_butyrate'] * Butyrate + 
                           p['k_scfa_propionate'] * Propionate + 
                           p['k_scfa_acetate'] * Acetate)
        
        # Diet influence on bacterial growth
        diet_factor_B = 1.0 + p['k_fiber_B'] * p['fiber_intake'] + p['k_protein_B'] * p['protein_intake']
        diet_factor_F = 1.0 + p['k_fiber_F'] * p['fiber_intake'] + p['k_protein_F'] * p['protein_intake'] + p['k_fat_F'] * p['fat_intake']
        diet_factor_Bi = 1.0 + p['k_fiber_Bi'] * p['fiber_intake']
        diet_factor_P = 1.0 + p['k_fiber_P'] * p['fiber_intake'] + p['k_protein_P'] * p['protein_intake'] + p['k_fat_P'] * p['fat_intake']
        diet_factors = np.array([diet_factor_B, diet_factor_F, diet_factor_Bi, diet_factor_P])
        
        # Antibiotic effect
        abx_effect = p['abx_level'] * np.array([p['k_abx_B'], p['k_abx_F'], p['k_abx_Bi'], p['k_abx_P']])
        
        # Growth rates adjusted for diet
        growth_rates = np.array([p['r_bacteroides'], p['r_firmicutes'], p['r_bifidobacteria'], p['r_proteobacteria']]) * diet_factors
        
        # Carrying capacities
        carrying_capacities = np.array([p['K_bacteroides'], p['K_firmicutes'], p['K_bifidobacteria'], p['K_proteobacteria']])
        
        # Calculate hormonal dynamics
        dGLP1 = (p['k_secr_glp1'] * (1 - GLP1/p['k_max_glp1']) * (1 + 0.7 * total_scfa_effect) -
                p['k_deg_glp1'] * GLP1 * bistable_factor(GLP1, p['bistable_threshold'], p['feedback_strength']) +
                p['k_vagus_glp1'] * Vagus)
        
        dGhrelin = (p['k_ghrelin'] * (1 - Ghrelin/p['k_max_ghrelin']) -
                   p['k_sat_ghrelin'] * GLP1 * Ghrelin * bistable_factor(Ghrelin, 0.5, 0.8))
        
        # Calculate neural dynamics
        food_intake = np.mean([p['fiber_intake'], p['protein_intake'], p['fat_intake']])
        food_distension = food_intake * 0.8  # Simplified distension model
        dVagus = (p['k_dist'] * food_distension -
                 p['k_deg_vagus'] * Vagus +
                 p['k_stress'] * p['stress_level'])
        
        # Calculate microbiome dynamics (Lotka-Volterra)
        # dB/dt = r*B*(1 - Σ(α*B_j)/K) - abx_effect*B
        bacterial_abundances = bacteria
        
        # Calculate the interaction terms
        interaction_terms = np.zeros(4)
        for i in range(4):
            interaction_terms[i] = np.sum(self.alpha_matrix[i] * bacterial_abundances) / carrying_capacities[i]
            
        # Calculate bacterial growth rates
        dBacteria = growth_rates * bacterial_abundances * (1 - interaction_terms) - abx_effect * bacterial_abundances
        
        # Calculate metabolite dynamics
        dButyrate = (p['k_butyrate_F'] * Firmicutes + p['k_butyrate_Bi'] * Bifidobacteria - 
                    p['k_deg_scfa'] * Butyrate)
        
        dPropionate = (p['k_propionate_B'] * Bacteroides - 
                      p['k_deg_scfa'] * Propionate)
        
        dAcetate = (p['k_acetate_B'] * Bacteroides + p['k_acetate_F'] * Firmicutes + 
                   p['k_acetate_Bi'] * Bifidobacteria - 
                   p['k_deg_scfa'] * Acetate)
        
        # Calculate energy balance
        dEnergy = (p['k_energy_intake'] * food_intake * (1 - 0.2 * GLP1) -
                  p['k_energy_expend'] * (1 + 0.3 * Ghrelin))
        
        # Combine all drift terms
        drift = np.concatenate(([dGLP1, dGhrelin, dVagus], dBacteria, [dButyrate, dPropionate, dAcetate, dEnergy]))
        
        return drift
    
    def diffusion_terms(self, t, y):
        """
        Calculate stochastic diffusion terms for the SDEs
        
        Args:
            t: time
            y: state vector
            
        Returns:
            Array of diffusion coefficients for each state variable
        """
        # Extract state variables and ensure non-negativity
        y = np.maximum(y, 0)
        
        GLP1, Ghrelin, Vagus = y[0:3]
        bacteria = y[3:7]
        metabolites = y[7:10]
        Energy = y[10]
        
        # Extract parameters
        p = self.params
        
        # State-dependent noise (multiplicative)
        sigma_GLP1 = p['sigma_glp1'] * GLP1
        sigma_Ghrelin = p['sigma_ghrelin'] * Ghrelin
        sigma_Vagus = p['sigma_vagus'] * Vagus
        
        # Bacterial noise (stronger during low abundance to model stochastic extinction)
        # This creates a floor on noise for low abundances
        sigma_bacteria = p['sigma_bacteria'] * (bacteria + 0.05)
        
        # Metabolite noise
        sigma_metabolites = p['sigma_metabolites'] * metabolites
        
        # Energy noise
        sigma_Energy = p['sigma_energy'] * Energy
        
        # Combine all diffusion terms
        diffusion = np.concatenate(([sigma_GLP1, sigma_Ghrelin, sigma_Vagus], 
                                   sigma_bacteria, 
                                   sigma_metabolites, 
                                   [sigma_Energy]))
        
        return diffusion
    
    def euler_maruyama_step(self, t, y, dt):
        """
        Single step of Euler-Maruyama method for SDE integration
        
        Args:
            t: current time
            y: current state
            dt: time step
            
        Returns:
            Updated state after one time step
        """
        drift = self.drift_terms(t, y)
        diffusion = self.diffusion_terms(t, y)
        
        # Generate Wiener increments
        dW = np.random.normal(0, np.sqrt(dt), size=len(y))
        
        # Euler-Maruyama update
        y_next = y + drift * dt + diffusion * dW
        
        # Ensure non-negativity
        y_next = np.maximum(y_next, 0)
        
        return y_next
    
    def simulate_sde(self, t_span, y0=None, dt=0.01, seed=None):
        """
        Simulate the system using Euler-Maruyama method
        
        Args:
            t_span: (start_time, end_time) tuple
            y0: initial state vector 
            dt: time step size
            seed: random seed for reproducibility
            
        Returns:
            t_values, y_values arrays
        """
        if seed is not None:
            np.random.seed(seed)
            
        if y0 is None:
            # Default initial conditions
            y0 = np.array([
                0.2,    # GLP1
                0.8,    # Ghrelin
                0.5,    # Vagus
                0.6,    # Bacteroides
                0.5,    # Firmicutes
                0.3,    # Bifidobacteria
                0.1,    # Proteobacteria
                0.2,    # Butyrate
                0.3,    # Propionate
                0.4,    # Acetate
                0.4     # Energy
            ])
        
        # Create time points
        t_start, t_end = t_span
        t_values = np.arange(t_start, t_end + dt, dt)
        n_steps = len(t_values)
        
        # Initialize state array
        y_values = np.zeros((n_steps, len(y0)))
        y_values[0] = y0
        
        # Integration loop
        for i in range(1, n_steps):
            y_values[i] = self.euler_maruyama_step(t_values[i-1], y_values[i-1], dt)
        
        return t_values, y_values
    
    def visualize_simulation(self, t, y, title="Microbiome-Gut-Brain Axis Simulation"):
        """
        Visualize simulation results
        
        Args:
            t: time array
            y: state array
            title: plot title
            
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Define subplot grid
        gs = plt.GridSpec(3, 3, figure=fig)
        
        # Hormonal dynamics
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.plot(t, y[:, 0], label='GLP-1')
        ax1.plot(t, y[:, 1], label='Ghrelin')
        ax1.plot(t, y[:, 2], label='Vagus')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Host Signaling')
        ax1.legend()
        ax1.grid(True)
        
        # Microbiome dynamics
        ax2 = fig.add_subplot(gs[1, 0:2])
        ax2.plot(t, y[:, 3], label='Bacteroides', color='blue')
        ax2.plot(t, y[:, 4], label='Firmicutes', color='red')
        ax2.plot(t, y[:, 5], label='Bifidobacteria', color='green')
        ax2.plot(t, y[:, 6], label='Proteobacteria', color='purple')
        ax2.set_ylabel('Relative Abundance')
        ax2.set_title('Microbiome Composition')
        ax2.legend()
        ax2.grid(True)
        
        # SCFA dynamics
        ax3 = fig.add_subplot(gs[2, 0:2])
        ax3.plot(t, y[:, 7], label='Butyrate', color='darkred')
        ax3.plot(t, y[:, 8], label='Propionate', color='darkorange')
        ax3.plot(t, y[:, 9], label='Acetate', color='darkgreen')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Concentration')
        ax3.set_title('Microbial Metabolites')
        ax3.legend()
        ax3.grid(True)
        
        # Energy balance
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.plot(t, y[:, 10], label='Energy', color='black')
        ax4.set_ylabel('Energy Level')
        ax4.set_title('Energy Homeostasis')
        ax4.grid(True)
        
        # Firmicutes/Bacteroides ratio
        ax5 = fig.add_subplot(gs[1, 2])
        fb_ratio = y[:, 4] / (y[:, 3] + 1e-6)  # Add small epsilon to avoid division by zero
        ax5.plot(t, fb_ratio, color='magenta')
        ax5.set_ylabel('F/B Ratio')
        ax5.set_title('Firmicutes/Bacteroides Ratio')
        ax5.grid(True)
        
        # Ecological metrics
        ax6 = fig.add_subplot(gs[2, 2])
        # Calculate Shannon diversity index
        def shannon_diversity(abundances):
            abundances = abundances / (np.sum(abundances) + 1e-10)
            return -np.sum(abundances * np.log(abundances + 1e-10))
        
        shannon = [shannon_diversity(y[i, 3:7]) for i in range(len(t))]
        ax6.plot(t, shannon, color='darkblue')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('Shannon Index')
        ax6.set_title('Microbial Diversity')
        ax6.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        return fig
    
    def visualize_microbiome_interaction_network(self):
        """Visualize the microbiome interaction network as a directed graph"""
        try:
            import networkx as nx
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            species = ['Bacteroides', 'Firmicutes', 'Bifidobacteria', 'Proteobacteria']
            for sp in species:
                G.add_node(sp)
            
            # Add edges with weights from interaction matrix
            for i, sp_i in enumerate(species):
                for j, sp_j in enumerate(species):
                    if i != j:  # Skip self-interactions
                        interaction = self.alpha_matrix[i, j]
                        if interaction != 0:
                            G.add_edge(sp_j, sp_i, weight=interaction)
            
            # Set up the plot
            plt.figure(figsize=(10, 8))
            
            # Define node positions
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
            
            # Draw edges with different colors for positive/negative interactions
            edges_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0]
            edges_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0]
            
            nx.draw_networkx_edges(G, pos, edgelist=edges_pos, width=2, edge_color='green', 
                                 arrowsize=20, arrowstyle='->')
            nx.draw_networkx_edges(G, pos, edgelist=edges_neg, width=2, edge_color='red', 
                                 arrowsize=20, arrowstyle='->')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
            
            # Add edge labels (weights)
            edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
            
            plt.axis('off')
            plt.title('Microbiome Interaction Network\nGreen: Facilitation, Red: Competition', fontsize=14)
            plt.tight_layout()
            
            return plt.gcf()
            
        except ImportError:
            print("NetworkX library required for interaction network visualization")
            return None
    
    def simulate_antibiotic_perturbation(self, antibiotic_start=50, antibiotic_duration=20, recovery_duration=100, 
                                       antibiotic_strength=0.7, dt=0.05):
        """
        Simulate antibiotic perturbation and recovery
        
        Args:
            antibiotic_start: time when antibiotic treatment starts
            antibiotic_duration: duration of antibiotic treatment
            recovery_duration: duration of recovery period
            antibiotic_strength: strength of antibiotic effect (0-1)
            dt: time step
            
        Returns:
            Simulation results
        """
        # Total simulation time
        t_end = antibiotic_start + antibiotic_duration + recovery_duration
        
        # Create antibiotic time profile
        time_points = np.arange(0, t_end + dt, dt)
        abx_profile = np.zeros_like(time_points)
        
        # Set antibiotic levels during treatment period
        treatment_start_idx = int(antibiotic_start / dt)
        treatment_end_idx = int((antibiotic_start + antibiotic_duration) / dt)
        abx_profile[treatment_start_idx:treatment_end_idx] = antibiotic_strength
        
        # Save original antibiotic level
        original_abx = self.params['abx_level']
        
        # Initialize results
        t_values = time_points
        n_steps = len(t_values)
        y0 = np.array([
            0.2,    # GLP1
            0.8,    # Ghrelin
            0.5,    # Vagus
            0.6,    # Bacteroides
            0.5,    # Firmicutes
            0.3,    # Bifidobacteria
            0.1,    # Proteobacteria
            0.2,    # Butyrate
            0.3,    # Propionate
            0.4,    # Acetate
            0.4     # Energy
        ])
        y_values = np.zeros((n_steps, len(y0)))
        y_values[0] = y0
        
        # Integration loop with time-varying antibiotic level
        for i in range(1, n_steps):
            # Update antibiotic level
            self.params['abx_level'] = abx_profile[i-1]
            
            # Take integration step
            y_values[i] = self.euler_maruyama_step(t_values[i-1], y_values[i-1], dt)
        
        # Restore original parameter
        self.params['abx_level'] = original_abx
        
        # Add antibiotic profile to results
        results = {
            't': t_values,
            'y': y_values,
            'abx_profile': abx_profile
        }
        
        return results
    
    def visualize_antibiotic_experiment(self, results, title="Antibiotic Perturbation and Recovery"):
        """
        Visualize results of antibiotic perturbation experiment
        
        Args:
            results: output from simulate_antibiotic_perturbation()
            title: plot title
            
        Returns:
            matplotlib figure
        """
        t = results['t']
        y = results['y']
        abx_profile = results['abx_profile']
        
        fig = plt.figure(figsize=(18, 14))
        
        # Define subplot grid
        gs = plt.GridSpec(4, 3, figure=fig)
        
        # Antibiotic profile
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, abx_profile, 'k-', label='Antibiotic')
        ax1.set_ylabel('Antibiotic Level')
        ax1.set_title('Antibiotic Profile')
        ax1.grid(True)
        
        # Microbiome dynamics
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(t, y[:, 3], label='Bacteroides', color='blue')
        ax2.plot(t, y[:, 4], label='Firmicutes', color='red')
        ax2.plot(t, y[:, 5], label='Bifidobacteria', color='green')
        ax2.plot(t, y[:, 6], label='Proteobacteria', color='purple')
        ax2.set_ylabel('Relative Abundance')
        ax2.set_title('Microbiome Composition')
        ax2.legend()
        ax2.grid(True)
        
        # SCFA dynamics
        ax3 = fig.add_subplot(gs[2, 0:2])
        ax3.plot(t, y[:, 7], label='Butyrate', color='darkred')
        ax3.plot(t, y[:, 8], label='Propionate', color='darkorange')
        ax3.plot(t, y[:, 9], label='Acetate', color='darkgreen')
        ax3.set_ylabel('Concentration')
        ax3.set_title('Microbial Metabolites')
        ax3.legend()
        ax3.grid(True)
        
        # Hormonal dynamics
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.plot(t, y[:, 0], label='GLP-1')
        ax4.plot(t, y[:, 1], label='Ghrelin')
        ax4.set_ylabel('Concentration')
        ax4.set_title('Host Hormones')
        ax4.legend()
        ax4.grid(True)
        
        # Ecological metrics
        ax5 = fig.add_subplot(gs[3, 0])
        # Calculate Shannon diversity index
        def shannon_diversity(abundances):
            abundances = abundances / (np.sum(abundances) + 1e-10)
            return -np.sum(abundances * np.log(abundances + 1e-10))
        
        shannon = [shannon_diversity(y[i, 3:7]) for i in range(len(t))]
        ax5.plot(t, shannon, color='darkblue')
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Shannon Index')
        ax5.set_title('Microbial Diversity')
        ax5.grid(True)
        
        # Firmicutes/Bacteroides ratio
        ax6 = fig.add_subplot(gs[3, 1])
        fb_ratio = y[:, 4] / (y[:, 3] + 1e-6)  # Add small epsilon to avoid division by zero
        ax6.plot(t, fb_ratio, color='magenta')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('F/B Ratio')
        ax6.set_title('Firmicutes/Bacteroides Ratio')
        ax6.grid(True)
        
        # Energy dynamics
        ax7 = fig.add_subplot(gs[3, 2])
        ax7.plot(t, y[:, 10], color='black')
        ax7.set_xlabel('Time (hours)')
        ax7.set_ylabel('Energy Level')
        ax7.set_title('Energy Homeostasis')
        ax7.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        return fig
    
    def visualize_taxonomy_heatmap(self, results, time_points=None, title="Taxonomic Composition Over Time"):
        """
        Create a heatmap showing the relative abundance of bacterial taxa over time
        
        Args:
            results: simulation results with time series data
            time_points: specific time points to sample (if None, samples evenly)
            title: plot title
            
        Returns:
            matplotlib figure
        """
        t = results['t']
        y = results['y']
        
        # Get bacterial abundances
        bacteria_data = y[:, 3:7]
        
        # Normalize to relative abundance
        total_abundance = np.sum(bacteria_data, axis=1, keepdims=True)
        relative_abundance = bacteria_data / (total_abundance + 1e-10)
        
        # Select time points
        if time_points is None:
            # Sample 20 time points evenly
            n_samples = 20
            idx = np.linspace(0, len(t)-1, n_samples, dtype=int)
        else:
            # Find closest time points
            idx = [np.abs(t - tp).argmin() for tp in time_points]
        
        # Extract data for selected time points
        sample_times = t[idx]
        sample_data = relative_abundance[idx, :]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create custom colormap (white to blue)
        cmap = LinearSegmentedColormap.from_list('abundance', ['white', 'blue'], N=256)
        
        # Plot heatmap
        im = ax.imshow(sample_data, aspect='auto', cmap=cmap, vmin=0, vmax=1)
        
        # Set y-axis labels (time points)
        ax.set_yticks(np.arange(len(sample_times)))
        ax.set_yticklabels([f't={time:.1f}' for time in sample_times])
        
        # Set x-axis labels (taxa)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(['Bacteroides', 'Firmicutes', 'Bifidobacteria', 'Proteobacteria'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Relative Abundance')
        
        # Add title
        ax.set_title(title)
        
        # If we have antibiotic data, mark the treatment period
        if 'abx_profile' in results:
            abx_profile = results['abx_profile']
            treatment_periods = []
            in_treatment = False
            start_idx = 0
            
            # Find treatment periods
            for i, level in enumerate(abx_profile):
                if not in_treatment and level > 0:
                    in_treatment = True
                    start_idx = i
                elif in_treatment and level == 0:
                    in_treatment = False
                    treatment_periods.append((start_idx, i))
            
            # If still in treatment at the end
            if in_treatment:
                treatment_periods.append((start_idx, len(abx_profile)-1))
            
            # Mark treatment periods on the y-axis
            for start, end in treatment_periods:
                # Convert to sampled time indices
                for i, idx_val in enumerate(idx):
                    if start <= idx_val <= end:
                        ax.get_yticklabels()[i].set_color('red')
        
        plt.tight_layout()
        
        return fig
    
    def simulate_diet_shifts(self, diet_changes, duration=50, dt=0.05):
        """
        Simulate the effect of diet shifts on the microbiome and host
        
        Args:
            diet_changes: list of dictionaries with diet parameters and time points
                e.g. [{'time': 0, 'fiber': 0.2, 'protein': 0.5, 'fat': 0.7},
                      {'time': 100, 'fiber': 0.8, 'protein': 0.3, 'fat': 0.2}]
            duration: duration to simulate after the last diet change
            dt: time step
            
        Returns:
            Simulation results
        """
        # Sort diet changes by time
        diet_changes = sorted(diet_changes, key=lambda x: x['time'])
        
        # Calculate total simulation time
        t_end = diet_changes[-1]['time'] + duration
        
        # Create time points
        time_points = np.arange(0, t_end + dt, dt)
        n_steps = len(time_points)
        
        # Create diet profiles
        fiber_profile = np.zeros_like(time_points)
        protein_profile = np.zeros_like(time_points)
        fat_profile = np.zeros_like(time_points)
        
        # Fill diet profiles
        for i in range(len(diet_changes)):
            current = diet_changes[i]
            
            # Determine end index
            if i < len(diet_changes) - 1:
                end_idx = int(diet_changes[i+1]['time'] / dt)
            else:
                end_idx = n_steps
            
            # Set diet values for this period
            start_idx = int(current['time'] / dt)
            fiber_profile[start_idx:end_idx] = current.get('fiber', self.params['fiber_intake'])
            protein_profile[start_idx:end_idx] = current.get('protein', self.params['protein_intake'])
            fat_profile[start_idx:end_idx] = current.get('fat', self.params['fat_intake'])
        
        # Save original diet parameters
        original_fiber = self.params['fiber_intake']
        original_protein = self.params['protein_intake']
        original_fat = self.params['fat_intake']
        
        # Initialize results
        t_values = time_points
        y0 = np.array([
            0.2,    # GLP1
            0.8,    # Ghrelin
            0.5,    # Vagus
            0.6,    # Bacteroides
            0.5,    # Firmicutes
            0.3,    # Bifidobacteria
            0.1,    # Proteobacteria
            0.2,    # Butyrate
            0.3,    # Propionate
            0.4,    # Acetate
            0.4     # Energy
        ])
        y_values = np.zeros((n_steps, len(y0)))
        y_values[0] = y0
        
        # Integration loop with time-varying diet
        for i in range(1, n_steps):
            # Update diet parameters
            self.params['fiber_intake'] = fiber_profile[i-1]
            self.params['protein_intake'] = protein_profile[i-1]
            self.params['fat_intake'] = fat_profile[i-1]
            
            # Take integration step
            y_values[i] = self.euler_maruyama_step(t_values[i-1], y_values[i-1], dt)
        
        # Restore original parameters
        self.params['fiber_intake'] = original_fiber
        self.params['protein_intake'] = original_protein
        self.params['fat_intake'] = original_fat
        
        # Add diet profiles to results
        results = {
            't': t_values,
            'y': y_values,
            'fiber_profile': fiber_profile,
            'protein_profile': protein_profile,
            'fat_profile': fat_profile
        }
        
        return results
    
    def visualize_diet_experiment(self, results, title="Effect of Diet Shifts on the Gut-Brain Axis"):
        """
        Visualize results of diet shift experiment
        
        Args:
            results: output from simulate_diet_shifts()
            title: plot title
            
        Returns:
            matplotlib figure
        """
        t = results['t']
        y = results['y']
        fiber_profile = results['fiber_profile']
        protein_profile = results['protein_profile']
        fat_profile = results['fat_profile']
        
        fig = plt.figure(figsize=(18, 14))
        
        # Define subplot grid
        gs = plt.GridSpec(4, 3, figure=fig)
        
        # Diet profiles
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, fiber_profile, 'g-', label='Fiber')
        ax1.plot(t, protein_profile, 'b-', label='Protein')
        ax1.plot(t, fat_profile, 'r-', label='Fat')
        ax1.set_ylabel('Intake Level')
        ax1.set_title('Diet Composition')
        ax1.legend()
        ax1.grid(True)
        
        # Microbiome dynamics
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(t, y[:, 3], label='Bacteroides', color='blue')
        ax2.plot(t, y[:, 4], label='Firmicutes', color='red')
        ax2.plot(t, y[:, 5], label='Bifidobacteria', color='green')
        ax2.plot(t, y[:, 6], label='Proteobacteria', color='purple')
        ax2.set_ylabel('Relative Abundance')
        ax2.set_title('Microbiome Composition')
        ax2.legend()
        ax2.grid(True)
        
        # SCFA dynamics
        ax3 = fig.add_subplot(gs[2, 0:2])
        ax3.plot(t, y[:, 7], label='Butyrate', color='darkred')
        ax3.plot(t, y[:, 8], label='Propionate', color='darkorange')
        ax3.plot(t, y[:, 9], label='Acetate', color='darkgreen')
        ax3.set_ylabel('Concentration')
        ax3.set_title('Microbial Metabolites')
        ax3.legend()
        ax3.grid(True)
        
        # Hormonal dynamics
        ax4 = fig.add_subplot(gs[2, 2])
        ax4.plot(t, y[:, 0], label='GLP-1')
        ax4.plot(t, y[:, 1], label='Ghrelin')
        ax4.set_ylabel('Concentration')
        ax4.set_title('Host Hormones')
        ax4.legend()
        ax4.grid(True)
        
        # Ecological metrics
        ax5 = fig.add_subplot(gs[3, 0])
        # Calculate Shannon diversity index
        def shannon_diversity(abundances):
            abundances = abundances / (np.sum(abundances) + 1e-10)
            return -np.sum(abundances * np.log(abundances + 1e-10))
        
        shannon = [shannon_diversity(y[i, 3:7]) for i in range(len(t))]
        ax5.plot(t, shannon, color='darkblue')
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Shannon Index')
        ax5.set_title('Microbial Diversity')
        ax5.grid(True)
        
        # Firmicutes/Bacteroides ratio
        ax6 = fig.add_subplot(gs[3, 1])
        fb_ratio = y[:, 4] / (y[:, 3] + 1e-6)
        ax6.plot(t, fb_ratio, color='magenta')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('F/B Ratio')
        ax6.set_title('Firmicutes/Bacteroides Ratio')
        ax6.grid(True)
        
        # Energy dynamics
        ax7 = fig.add_subplot(gs[3, 2])
        ax7.plot(t, y[:, 10], color='black')
        ax7.set_xlabel('Time (hours)')
        ax7.set_ylabel('Energy Level')
        ax7.set_title('Energy Homeostasis')
        ax7.grid(True)
        
        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.93)
        
        return fig


# Example usage

# Create model with specified parameters
model = MicrobiomeGBAModel({
    'fiber_intake': 0.6,
    'sigma_bacteria': 0.05
})

# Run standard simulation
t, y = model.simulate_sde(t_span=(0, 200), dt=0.05, seed=42)

# Visualize results
fig1 = model.visualize_simulation(t, y, title="Baseline Microbiome-Gut-Brain Axis Dynamics")

# Visualize microbiome interaction network
fig2 = model.visualize_microbiome_interaction_network()

# Simulate antibiotic perturbation
abx_results = model.simulate_antibiotic_perturbation(
    antibiotic_start=50,
    antibiotic_duration=30,
    recovery_duration=150,
    antibiotic_strength=0.8
)

# Visualize antibiotic experiment
fig3 = model.visualize_antibiotic_experiment(abx_results)
fig4 = model.visualize_taxonomy_heatmap(abx_results)

# Simulate diet shifts
diet_results = model.simulate_diet_shifts([
    {'time': 0, 'fiber': 0.2, 'protein': 0.6, 'fat': 0.7},   # Low-fiber, high-fat diet
    {'time': 100, 'fiber': 0.8, 'protein': 0.3, 'fat': 0.2}, # High-fiber, low-fat diet
    {'time': 200, 'fiber': 0.5, 'protein': 0.5, 'fat': 0.5}  # Balanced diet
], duration=100)

# Visualize diet experiment
fig5 = model.visualize_diet_experiment(diet_results)

plt.show()
