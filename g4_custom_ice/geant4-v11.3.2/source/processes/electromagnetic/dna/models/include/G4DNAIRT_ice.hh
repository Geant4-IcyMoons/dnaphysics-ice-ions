//
// ********************************************************************
// G4DNAIRT_ice.hh - Ice-specific Independent Reaction Times with Arrhenius diffusion
//
// Author: W. G. Shin (original)
// Modified for ice chemistry with Arrhenius diffusion coefficients
//
// Implements position-dependent Independent Reaction Time (IRT) algorithm using
// temperature-dependent diffusion coefficients. IRT calculates molecular encounter
// rates and reaction probabilities based on D = D₀ × exp(-Eₐ/kT) for OH/H molecules.
// ********************************************************************
//

#ifndef G4DNAIRT_ICE_HH_
#define G4DNAIRT_ICE_HH_

#include "G4DNAIRT.hh"

class G4MolecularConfiguration;

/**
 * G4DNAIRT_ice: Ice-specific Independent Reaction Time calculator
 * 
 * Handles molecular reaction timing using position-dependent Arrhenius
 * diffusion coefficients. Calculates when molecules will encounter
 * and react based on temperature-dependent mobility.
 * 
 * Arrhenius parameters:
 * - OH molecules: Eₐ=0.14 eV → 10³ change from 273K to 100K
 * - H molecules:  Eₐ=0.02 eV → 10¹ change from 273K to 100K
 * - Reaction rates ∝ √(D₁ + D₂) with dramatic spatial variation
 */
class G4DNAIRT_ice : public G4DNAIRT
{
public:
    G4DNAIRT_ice();
    explicit G4DNAIRT_ice(G4VDNAReactionModel*);
    ~G4DNAIRT_ice() override = default;
    G4DNAIRT_ice(const G4DNAIRT_ice& other) = delete;
    G4DNAIRT_ice& operator=(const G4DNAIRT_ice& other) = delete;

    // Main method: calculates encounter time using Arrhenius coefficients
    G4double GetIndependentReactionTime(const G4MolecularConfiguration*, const G4MolecularConfiguration*, G4double, const G4ThreeVector& position);
    
    // Executes molecular reaction with ice-specific positioning
    std::unique_ptr<G4ITReactionChange> MakeReaction(const G4Track&, const G4Track&) override;

private:
    // Helper: Get Arrhenius coefficient from track position
    G4double GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf);
    
    // Helper: Get Arrhenius coefficient from position and molecule type
    G4double GetArrheniusDiffusionCoefficientFromMolConf(const G4ThreeVector& position, const G4MolecularConfiguration* molConf);
};

#endif /* G4DNAIRT_ICE_HH_ */
