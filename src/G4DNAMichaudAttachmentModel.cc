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
//

// Created by Z. Francis (copy: Michaud variant)

#include "G4DNAMichaudAttachmentModel.hh"
#include "G4SystemOfUnits.hh"
#include "G4DNAChemistryManager.hh"
#include "G4DNAMolecularMaterial.hh"
#include "ModelDataRegistry.hh"

using namespace std;

G4DNAMichaudAttachmentModel::G4DNAMichaudAttachmentModel(const G4ParticleDefinition*,
                                                       const G4String& nam) :
G4VEmModel(nam) 
{
  fpWaterDensity = nullptr;

  SetLowEnergyLimit(4.*eV);
  SetHighEnergyLimit(13.*eV);

  verboseLevel = 0;
  // Verbosity scale:
  // 0 = nothing
  // 1 = warning for energy non-conservation
  // 2 = details of energy budget
  // 3 = calculation of cross sections, file openings, sampling of atoms
  // 4 = entering in methods

  fParticleChangeForGamma = nullptr;
  fDissociationFlag = true;
  fData = nullptr;
  statCode = false;
}

G4DNAMichaudAttachmentModel::~G4DNAMichaudAttachmentModel()
{
  delete fData;
}

void G4DNAMichaudAttachmentModel::Initialise(const G4ParticleDefinition* particle,
                                            const G4DataVector& /*cuts*/)
{
  if(particle->GetParticleName() != "e-")
  {
    G4Exception("G4DNAMichaudAttachmentModel::Initialise",
                "em0002",
                FatalException,
                "Model not applicable to particle type.");
  }

  if (LowEnergyLimit() < 4.*eV)
  {
    G4ExceptionDescription errMsg;
    errMsg << "G4DNAMichaudAttachmentModel: low energy limit increased from " <<
    LowEnergyLimit()/eV << " eV to " << 4.  << " eV" << G4endl;
    G4Exception("G4DNAMichaudAttachmentModel::Initialise",
                "Michaud_LowerEBoundary",
                JustWarning,
                errMsg);
    SetLowEnergyLimit(4*eV);
  }

  if (HighEnergyLimit() > 13.*eV)
  {
    G4ExceptionDescription errMsg;
    errMsg << "G4DNAMichaudAttachmentModel: high energy limit decreased from " <<
    HighEnergyLimit()/eV << " eV to " << 13. << " eV" << G4endl;
    G4Exception("G4DNAMichaudAttachmentModel::Initialise",
                "Michaud_HigherEBoundary",
                JustWarning,
                errMsg);
    SetHighEnergyLimit(13.*eV);
  }

  G4double scaleFactor = 1e-16*cm2;
  G4String fileElectron("dna/sigma_attachment_e_michaud");
  ModelDataRegistry::Instance().Record(
    std::string("model_ref:") + GetName(),
    ModelDataRegistry::NormalizeDatBasename(fileElectron));
  fData = new G4DNACrossSectionDataSet(new G4LogLogInterpolation(),
                                        eV, scaleFactor);
  fData->LoadData(fileElectron);

  fpWaterDensity = G4DNAMolecularMaterial::Instance()->
      GetNumMolPerVolTableFor(G4Material::GetMaterial("G4_WATER"));

  if (isInitialised) return;

  fParticleChangeForGamma = GetParticleChangeForGamma();
  isInitialised = true;
}

G4double
G4DNAMichaudAttachmentModel::CrossSectionPerVolume(const G4Material* material,
                                                const G4ParticleDefinition*,
                                                G4double ekin,
                                                G4double,
                                                G4double)
{
  G4double sigma = 0.;
  G4double waterDensity = (*fpWaterDensity)[material->GetIndex()];
  if (ekin >= LowEnergyLimit() && ekin <= HighEnergyLimit())
    sigma = fData->FindValue(ekin);
  return sigma*waterDensity;
}

void
G4DNAMichaudAttachmentModel::
SampleSecondaries(std::vector<G4DynamicParticle*>* /*fvect*/,
                  const G4MaterialCutsCouple* /*couple*/,
                  const G4DynamicParticle* aDynamicElectron,
                  G4double,
                  G4double)
{
  G4double electronEnergy0 = aDynamicElectron->GetKineticEnergy();
  if (!statCode)     
  {     
      fParticleChangeForGamma->SetProposedKineticEnergy(0.);
      fParticleChangeForGamma->ProposeTrackStatus(fStopAndKill);
      fParticleChangeForGamma->ProposeLocalEnergyDeposit(electronEnergy0);
  }
  else 
  {
      fParticleChangeForGamma->SetProposedKineticEnergy(electronEnergy0);
      fParticleChangeForGamma->ProposeLocalEnergyDeposit(electronEnergy0);
  }
  if(fDissociationFlag)
  {
    G4DNAChemistryManager::Instance()->
      CreateWaterMolecule(eDissociativeAttachment,
                          -1,
                          fParticleChangeForGamma->GetCurrentTrack());
  }
  return;
}
