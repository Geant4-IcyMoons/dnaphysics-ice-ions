//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// G4DNAMolecularDissociation_ice.cc - Ice-specific molecular dissociation with Arrhenius diffusion
// ---------------------------------------------------------------------
//  G4DNAMolecularDissociation_ice.cc
//  Customized by G. Yoffe for ice (solid H2O) chemistry
//
//  Implements molecular dissociation in ice, using Arrhenius diffusion
//  and ice-specific rates. Differs from the original by using temperature-
//  dependent coefficients and mechanisms for ice, not liquid water.
// ---------------------------------------------------------------------
//

#include "G4DNAMolecularDissociation_ice.hh"
#include "G4DNABrownianTransportation_ice.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4Molecule.hh"
#include "G4MoleculeTable.hh"
#include "G4MolecularConfiguration.hh"
#include "G4SystemOfUnits.hh"
#include "G4Hydrogen.hh"
#include "G4OH.hh"
#include "ArrheniusConstants_ice.hh"
#include <iomanip>

// DEBUG CONTROL: Set to true to enable creation prompts for OH and H molecules
static const G4bool g_debugMoleculeCreation = false; // Set to false to disable molecule creation debug output


// Helper function to get temperature from ice chemistry instance
// This accesses the temperature profile set in the macro
G4double GetTemperatureAtZ(G4double z_position) {
    // Use the global temperature function from the ice transport
    G4ThreeVector position(0, 0, z_position);
    return GetIceTemperatureAtPosition(position);
}

G4DNAMolecularDissociation_ice::G4DNAMolecularDissociation_ice(const G4String& processName)
    : G4DNAMolecularDissociation(processName)
{
    G4cout << "*** Ice-specific molecular dissociation with Arrhenius diffusion initialized ***" << G4endl;
}

G4VParticleChange* G4DNAMolecularDissociation_ice::DecayIt(const G4Track& track, const G4Step& step)
{
    // G4cout << "DEBUG: Ice-specific dissociation DecayIt called" << G4endl;
    
    // First, do the standard water dissociation
    G4VParticleChange* particleChange = G4DNAMolecularDissociation::DecayIt(track, step);
    
    // Set initial diffusion coefficients using Arrhenius relation for OH and H molecules
    G4int nbSecondaries = particleChange->GetNumberOfSecondaries();
    // G4cout << "DEBUG: Dissociation produced " << nbSecondaries << " secondary tracks" << G4endl;
    
    for (G4int i = 0; i < nbSecondaries; i++) {
        G4Track* secondaryTrack = particleChange->GetSecondary(i);
        if (secondaryTrack) {
            G4ParticleDefinition* particleDef = secondaryTrack->GetDefinition();
            G4ThreeVector position = secondaryTrack->GetPosition();
            G4double temperature = GetTemperatureAtZ(position.z());
            
            // Calculate Arrhenius diffusion coefficients using centralized constants
            if (particleDef == G4OH::Definition()) {
                // OH diffusion coefficient
                G4double D_OH = ArrheniusConstants_ice::CalculateArrheniusDiffusion(
                    ArrheniusConstants_ice::D0_OH, ArrheniusConstants_ice::Ea_OH, temperature);
                
                if (g_debugMoleculeCreation) {
                    G4cout << "*** OH MOLECULE CREATED ***" << G4endl;
                    G4cout << "  Parent Track ID: " << track.GetTrackID() << " (" << track.GetDefinition()->GetParticleName() << ")" << G4endl;
                    G4cout << "  Secondary Index: " << i << " (Track ID assigned later by tracking manager)" << G4endl;
                    G4cout << "  Position: z = " << std::fixed << std::setprecision(4) << position.z()/mm << " mm" << G4endl;
                    G4cout << "  Temperature: T = " << std::fixed << std::setprecision(1) << temperature/kelvin << " K" << G4endl;
                    G4cout << "  Arrhenius Coeff: D = " << std::scientific << std::setprecision(3) << D_OH/(m2/s) << " m²/s" << G4endl;
                    G4cout << "  Time: " << std::fixed << std::setprecision(2) << secondaryTrack->GetGlobalTime()/picosecond << " ps" << G4endl;
                    G4cout << "  Total Secondaries: " << nbSecondaries << " from this dissociation" << G4endl;
                    G4cout << "============================" << G4endl;
                }
            }
            else if (particleDef == G4Hydrogen::Definition()) {
                // H diffusion coefficient
                G4double D_H = ArrheniusConstants_ice::CalculateArrheniusDiffusion(
                    ArrheniusConstants_ice::D0_H, ArrheniusConstants_ice::Ea_H, temperature);
                
                if (g_debugMoleculeCreation) {
                    G4cout << "*** H ATOM CREATED ***" << G4endl;
                    G4cout << "  Parent Track ID: " << track.GetTrackID() << " (" << track.GetDefinition()->GetParticleName() << ")" << G4endl;
                    G4cout << "  Secondary Index: " << i << " (Track ID assigned later by tracking manager)" << G4endl;
                    G4cout << "  Position: z = " << std::fixed << std::setprecision(4) << position.z()/mm << " mm" << G4endl;
                    G4cout << "  Temperature: T = " << std::fixed << std::setprecision(1) << temperature/kelvin << " K" << G4endl;
                    G4cout << "  Arrhenius Coeff: D = " << std::scientific << std::setprecision(3) << D_H/(m2/s) << " m²/s" << G4endl;
                    G4cout << "  Time: " << std::fixed << std::setprecision(2) << secondaryTrack->GetGlobalTime()/picosecond << " ps" << G4endl;
                    G4cout << "  Total Secondaries: " << nbSecondaries << " from this dissociation" << G4endl;
                    G4cout << "===========================" << G4endl;
                }
            }
        }
    }
    
    return particleChange;
}

