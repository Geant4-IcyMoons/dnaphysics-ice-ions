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
// Michaud–ELSEPA blended elastic model (low-energy branch, 1.7–200 eV)
// Uses total cross sections from sigma_elastic_e_michaud_elsepa_low.dat
// and samples angles isotropically (as provided by the low-energy blend).
//

#ifndef G4DNAMichaud_ELSEPA_LOW_ELASTIC_hh
#define G4DNAMichaud_ELSEPA_LOW_ELASTIC_hh 1

#include "G4VEmModel.hh"
#include "G4DNACrossSectionDataSet.hh"
#include "G4ParticleChangeForGamma.hh"

class G4DNAMolecularMaterial;

class G4DNAMichaud_ELSEPA_LOW_ELASTIC : public G4VEmModel
{
 public:
  G4DNAMichaud_ELSEPA_LOW_ELASTIC(const G4ParticleDefinition* p = nullptr,
                                  const G4String& name = "G4DNAMichaud_ELSEPA_LOW_ELASTIC");
  ~G4DNAMichaud_ELSEPA_LOW_ELASTIC() override;

  void Initialise(const G4ParticleDefinition*, const G4DataVector&) override;
  G4double CrossSectionPerVolume(const G4Material* material,
                                 const G4ParticleDefinition* p,
                                 G4double ekin,
                                 G4double emin,
                                 G4double emax) override;
  void SampleSecondaries(std::vector<G4DynamicParticle*>*,
                         const G4MaterialCutsCouple*,
                         const G4DynamicParticle*,
                         G4double,
                         G4double) override;

 private:
  G4double RandomizeCosThetaIsotropic() const;

 private:
  G4ParticleChangeForGamma* fParticleChangeForGamma;
  const G4DNAMolecularMaterial* fpMolWaterDensity;
  G4DNACrossSectionDataSet* fpData;
};

#endif
