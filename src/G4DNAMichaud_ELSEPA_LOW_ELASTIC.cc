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
// Uses total cross sections from sigma_elastic_e_michaud_elsepa_low.dat.
// Angular distribution is isotropic (consistent with low-energy blend).
//

#include "G4DNAMichaud_ELSEPA_LOW_ELASTIC.hh"

#include "G4DNAMolecularMaterial.hh"
#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4DynamicParticle.hh"
#include "G4ThreeVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4Exp.hh"
#include "G4RandomTools.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4DNAMichaud_ELSEPA_LOW_ELASTIC::
G4DNAMichaud_ELSEPA_LOW_ELASTIC(const G4ParticleDefinition*, const G4String& nam)
: G4VEmModel(nam),
  fParticleChangeForGamma(nullptr),
  fpMolWaterDensity(nullptr),
  fpData(nullptr)
{
  SetLowEnergyLimit(1.7 * eV);
  SetHighEnergyLimit(200. * eV);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4DNAMichaud_ELSEPA_LOW_ELASTIC::~G4DNAMichaud_ELSEPA_LOW_ELASTIC()
{
  delete fpData;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4DNAMichaud_ELSEPA_LOW_ELASTIC::Initialise(const G4ParticleDefinition* particle,
                                                 const G4DataVector&)
{
  if(particle->GetParticleName() != "e-")
  {
    G4Exception("G4DNAMichaud_ELSEPA_LOW_ELASTIC::Initialise",
                "em0002",
                FatalException,
                "Model not applicable to particle type.");
  }

  if (LowEnergyLimit() < 1.7 * eV)
  {
    SetLowEnergyLimit(1.7 * eV);
  }
  if (HighEnergyLimit() > 200. * eV)
  {
    SetHighEnergyLimit(200. * eV);
  }

  if (isInitialised) { return; }

  // Total cross section
  const G4double scaleFactor = 1e-16 * cm * cm;
  const G4String fileElectron("dna/sigma_elastic_e_michaud_elsepa_low");
  fpData = new G4DNACrossSectionDataSet(new G4LogLogInterpolation(), eV, scaleFactor);
  fpData->LoadData(fileElectron);

  // Water density
  G4DNAMolecularMaterial::Instance()->Initialize();
  fpMolWaterDensity = G4DNAMolecularMaterial::Instance()->
    GetNumMolPerVolTableFor(G4Material::GetMaterial("G4_WATER"));

  fParticleChangeForGamma = GetParticleChangeForGamma();
  isInitialised = true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double
G4DNAMichaud_ELSEPA_LOW_ELASTIC::
CrossSectionPerVolume(const G4Material* material,
                      const G4ParticleDefinition*,
                      G4double ekin,
                      G4double,
                      G4double)
{
  G4double sigma = 0.;
  const G4double waterDensity = (*fpMolWaterDensity)[material->GetIndex()];

  if (ekin <= HighEnergyLimit() && ekin >= LowEnergyLimit())
  {
    sigma = fpData->FindValue(ekin);
  }

  return sigma * waterDensity;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void G4DNAMichaud_ELSEPA_LOW_ELASTIC::
SampleSecondaries(std::vector<G4DynamicParticle*>*,
                  const G4MaterialCutsCouple*,
                  const G4DynamicParticle* aDynamicElectron,
                  G4double,
                  G4double)
{
  const G4double electronEnergy0 = aDynamicElectron->GetKineticEnergy();

  const G4double cosTheta = RandomizeCosThetaIsotropic();
  const G4double phi = 2. * pi * G4UniformRand();

  const G4ThreeVector zVers = aDynamicElectron->GetMomentumDirection();
  const G4ThreeVector xVers = zVers.orthogonal();
  const G4ThreeVector yVers = zVers.cross(xVers);

  G4double xDir = std::sqrt(1. - cosTheta*cosTheta);
  G4double yDir = xDir;
  xDir *= std::cos(phi);
  yDir *= std::sin(phi);

  const G4ThreeVector zPrimeVers((xDir*xVers + yDir*yVers + cosTheta*zVers));

  fParticleChangeForGamma->ProposeMomentumDirection(zPrimeVers.unit());
  fParticleChangeForGamma->SetProposedKineticEnergy(electronEnergy0);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

G4double G4DNAMichaud_ELSEPA_LOW_ELASTIC::RandomizeCosThetaIsotropic() const
{
  // Isotropic: cos(theta) uniformly in [-1,1]
  return 2.0 * G4UniformRand() - 1.0;
}
