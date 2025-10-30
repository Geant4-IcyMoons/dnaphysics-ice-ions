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
#ifndef G4EmDNAChemistry_ice_hh
#define G4EmDNAChemistry_ice_hh 1

#include "G4VPhysicsConstructor.hh"
#include "G4VUserChemistryList.hh"
#include "G4ChemDissociationChannels_ice.hh"
#include "globals.hh"
#include <vector>

class G4DNAMolecularReactionTable;

// Forward declaration for custom reaction data
class G4DNAPositionDependentReactionData;

class G4EmDNAChemistry_ice : public G4VUserChemistryList, public G4VPhysicsConstructor
{
  public:
    G4EmDNAChemistry_ice();
    ~G4EmDNAChemistry_ice() override = default;

    void ConstructParticle() override { ConstructMolecule(); }
    void ConstructMolecule() override;
    void ConstructProcess() override;

    void ConstructDissociationChannels() override;
    void ConstructReactionTable(G4DNAMolecularReactionTable* reactionTable) override;
    void ConstructTimeStepModel(G4DNAMolecularReactionTable* reactionTable) override;

    // Methods to set temperature and depth profiles
    void SetTemperatureProfile(const std::vector<G4double>& temperatures);
    void SetDepthProfile(const std::vector<G4double>& depths);
    
    // Method to set ice phase
    void SetIceAmorphous(G4bool amorphous);
    
    // Access methods
    const std::vector<G4double>& GetTemperatures() const { return fTemperatures; }
    const std::vector<G4double>& GetDepths() const { return fDepths; }
    G4bool GetIceAmorphous() const { return fIceAmorphous; }
    
    // Get temperature at specific z-position
    G4double GetTemperatureAtPosition(G4double z_position) const;

  private:
    std::vector<G4double> fTemperatures;  // Temperature values [K]
    std::vector<G4double> fDepths;        // Depth boundaries [mm]
    G4bool fIceAmorphous;                 // Ice phase: true = amorphous, false = crystalline
    
    // Debug function for Arrhenius temperature-dependent diffusion
    void DebugArrheniusSetup();
};

#endif
