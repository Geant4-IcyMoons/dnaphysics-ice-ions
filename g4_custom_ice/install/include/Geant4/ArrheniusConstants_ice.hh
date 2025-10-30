#ifndef ArrheniusConstants_ice_hh
#define ArrheniusConstants_ice_hh 1

#include "G4SystemOfUnits.hh"
#include <cmath> // For std::exp

// Centralized Arrhenius diffusion parameters for ice chemistry
namespace ArrheniusConstants_ice {
    
    // OH diffusion parameters [Miyazaki+22]
    static const G4double D0_OH = 9e-8 * (m2/s);           // Pre-exponential factor for OH [Miyazaki+22]
    static const G4double Ea_OH = 0.14 * eV;                // Activation energy for OH diffusion [Miyazaki+22]
    
    // H diffusion parameters [Watanabe+10]
    static const G4double D0_H = 9e-8 * (m2/s);            // Pre-exponential factor for H
    static const G4double Ea_H = 0.02 * eV;                 // Activation energy for H diffusion [Watanabe+10]
    
    // Boltzmann constant in eV/K
    static const G4double kB_eV = 8.617333e-5 * eV / kelvin;
    
    // Helper function to calculate Arrhenius diffusion coefficient
    inline G4double CalculateArrheniusDiffusion(G4double D0, G4double Ea, G4double temperature) {
        return D0 * std::exp(-Ea / (kB_eV * temperature));
    }
}

#endif 