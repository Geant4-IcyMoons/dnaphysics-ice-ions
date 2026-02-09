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
// Based on the work described in
// Rad Res 163, 98-111 (2005)
// D. Emfietzoglou_ice, H. Nikjoo
// 
// Authors of the class (2014):
// I. Kyriakou (kyriak@cc.uoi.gr)
// D. Emfietzoglou_ice (demfietz@cc.uoi.gr)
// S. Incerti (incerti@cenbg.in2p3.fr)
//

#ifndef G4DNAEmfietzoglou_iceExcitationModel_h
#define G4DNAEmfietzoglou_iceExcitationModel_h 1

#include "G4VEmModel.hh"
#include "G4ParticleChangeForGamma.hh"
#include "G4ProductionCutsTable.hh"

#include "G4DNACrossSectionDataSet.hh"
#include "G4LogLogInterpolation.hh"
#include "G4Electron.hh"
#include "G4Proton.hh"
#include "G4DNAEmfietzoglou_iceExcitationStructure.hh"
#include "G4NistManager.hh"

class G4DNAEmfietzoglou_iceExcitationModel : public G4VEmModel
{

public:

  G4DNAEmfietzoglou_iceExcitationModel(const G4ParticleDefinition* p = nullptr,
                                   const G4String& nam =
                                       "DNAEmfietzoglou_iceExcitationModel");

  ~G4DNAEmfietzoglou_iceExcitationModel() override;

  G4DNAEmfietzoglou_iceExcitationModel & operator=(const G4DNAEmfietzoglou_iceExcitationModel &right) = delete;
  G4DNAEmfietzoglou_iceExcitationModel(const G4DNAEmfietzoglou_iceExcitationModel&) = delete;

  void Initialise(const G4ParticleDefinition*,
                          const G4DataVector& = *(new G4DataVector())) override;

  G4double CrossSectionPerVolume(const G4Material* material,
                                         const G4ParticleDefinition* p,
                                         G4double ekin,
                                         G4double emin,
                                         G4double emax) override;

  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
                                 const G4MaterialCutsCouple*,
                                 const G4DynamicParticle*,
                                 G4double tmin,
                                 G4double maxEnergy) override;

  inline void SelectStationary(G4bool input); 

  static G4int GetLastExcitationIndex();
  static void ClearLastExcitationIndex();
  static G4double GetLastPartialSigma_cm2();
  static void ClearLastPartialSigma_cm2();

protected:

  G4ParticleChangeForGamma* fParticleChangeForGamma;

private:

  G4bool statCode;

  // Water density table
  const std::vector<G4double>* fpMolWaterDensity;

  std::map<G4String, G4double, std::less<G4String> > lowEnergyLimit;
  std::map<G4String, G4double, std::less<G4String> > highEnergyLimit;

  G4bool isInitialised{false};
  G4int verboseLevel;

  // Cross section

  using MapFile = std::map<G4String, G4String, std::less<G4String>>;
  MapFile tableFile;

  using MapData = std::map<G4String, G4DNACrossSectionDataSet *, std::less<G4String>>;
  MapData tableData;

  // Partial cross section

  G4int RandomSelect(G4double energy, const G4String& particle);

  // Final state

  G4DNAEmfietzoglou_iceExcitationStructure waterStructure;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

inline void G4DNAEmfietzoglou_iceExcitationModel::SelectStationary (G4bool input)
{ 
    statCode = input; 
}		 

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

#endif
