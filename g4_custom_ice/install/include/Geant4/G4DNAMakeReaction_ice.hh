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
// G4DNAMakeReaction_ice.hh - Ice-specific molecular reaction maker with Arrhenius diffusion
//
// PURPOSE: This class handles the creation and positioning of molecular reactions
// in Europa's ice environment. Unlike vanilla G4DNAMakeReaction, it accounts for
// position-dependent diffusion rates when determining where reactions occur.
//
// ROLE IN GEANT4 CHEMISTRY:
// G4DNAMakeReaction is responsible for:
// 1. Creating reaction products at the correct spatial location
// 2. Updating molecular positions before/after reactions
// 3. Ensuring proper reaction kinematics based on diffusion physics
//
// ICE-SPECIFIC MODIFICATIONS:
// - MakeReaction(): Uses Arrhenius diffusion for reaction site calculation
// - UpdatePositionForReaction(): Updates molecular positions using ice-specific rates
// - Both methods now consider Europa's temperature gradients
//
// PHYSICS IMPACT:
// In vanilla Geant4, all reactions use fixed diffusion coefficients, creating
// uniform reaction distributions. In ice, slower diffusion in cold regions
// leads to different reaction patterns - this class ensures spatial realism.
//

#ifndef G4DNAMakeReaction_ice_hh
#define G4DNAMakeReaction_ice_hh 1

#include "G4DNAMakeReaction.hh"

class G4MolecularConfiguration;

/**
  * G4DNAMakeReaction_ice: Ice-specific molecular reaction maker
  * 
  * TECHNICAL DETAILS:
  * This class inherits from G4DNAMakeReaction and overrides key methods to use
  * position-dependent diffusion coefficients. The main algorithmic change is
  * replacing stored diffusion values with real-time Arrhenius calculations.
  * 
  * METHOD OVERRIDES:
  * 1. MakeReaction() - Creates reaction products using ice-specific diffusion
  * 2. UpdatePositionForReaction() - Updates molecular positions with ice physics
  * 
  * ARRHENIUS PARAMETERS:
  * - OH: D₀=9×10⁻⁸ m²/s, Eₐ=0.14 eV (temperature-sensitive)
  * - H:  D₀=9×10⁻⁸ m²/s, Eₐ=0.02 eV (less temperature-sensitive)
  * - Others: Use stored coefficients (backward compatibility)
  */
class G4DNAMakeReaction_ice : public G4DNAMakeReaction
{
public:
    // CONSTRUCTORS: Standard inheritance pattern
    G4DNAMakeReaction_ice();
    explicit G4DNAMakeReaction_ice(G4VDNAReactionModel*);
    ~G4DNAMakeReaction_ice() override = default;
    G4DNAMakeReaction_ice(const G4DNAMakeReaction_ice& other) = delete;
    G4DNAMakeReaction_ice& operator=(const G4DNAMakeReaction_ice& other) = delete;

    // MAIN OVERRIDES: Where ice physics is implemented
    // These methods replace vanilla behavior with Arrhenius-based calculations
    
    // Creates reaction products with ice-specific positioning
    // VANILLA DIFFERENCE: Uses stored diffusion coefficients
    // ICE VERSION: Uses Arrhenius coefficients based on current temperature
    std::unique_ptr<G4ITReactionChange> MakeReaction(const G4Track&, const G4Track&) override;
    
    // Updates molecular positions before reaction occurs
    // VANILLA DIFFERENCE: Position updates ignore temperature gradients
    // ICE VERSION: Position updates reflect actual diffusion rates in ice
    void UpdatePositionForReaction(G4Track&, G4Track&);

private:
    // ICE-SPECIFIC HELPER: Core innovation for position-dependent chemistry
    // Calculates Arrhenius diffusion coefficient based on track position
    // VANILLA EQUIVALENT: None - vanilla uses only stored values
    G4double GetArrheniusDiffusionCoefficient(const G4Track& track, const G4MolecularConfiguration* molConf);
};

#endif
