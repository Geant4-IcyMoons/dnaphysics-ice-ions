#include "PhysicsList_Proton.hh"

#include "G4EmDNAPhysics_option2.hh"
#include "G4EmParameters.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"

#include "G4DNAIonisation.hh"
#include "G4DNAExcitation.hh"
#include "G4DNARuddIonisationModel.hh"
#include "G4DNAMillerGreenExcitationModel.hh"
#include "G4DNAEmfietzoglou_iceProtonIonisationModel.hh"
#include "G4DNAEmfietzoglou_iceProtonExcitationModel.hh"
#include "G4Proton.hh"

PhysicsList_Proton::PhysicsList_Proton() : G4VModularPhysicsList()
{
  SetDefaultCutValue(1.0 * micrometer);
  SetVerboseLevel(1);

  // Keep Geant4-DNA particle definitions available.
  fEmPhysicsList = new G4EmDNAPhysics_option2();

  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(10.0 * eV, 1.0 * GeV);
  auto* param = G4EmParameters::Instance();
  param->SetMinEnergy(10.0 * eV);
  param->SetMaxEnergy(1.0 * GeV);
}

PhysicsList_Proton::~PhysicsList_Proton() = default;

void PhysicsList_Proton::ConstructParticle()
{
  fEmPhysicsList->ConstructParticle();
}

void PhysicsList_Proton::ConstructProcess()
{
  AddTransportation();

  auto* ph = G4PhysicsListHelper::GetPhysicsListHelper();

  // Proton excitation model tiling:
  // - Geant4-DNA water (Miller-Green): 10 eV - 100 keV
  // - Custom ice model:                100 keV - 10 MeV
  constexpr G4double kExcWaterLow = 10.0 * eV;
  constexpr G4double kExcWaterHigh = 100.0 * keV;
  constexpr G4double kExcIceLow = 100.0 * keV;
  constexpr G4double kExcIceHigh = 10.0 * MeV;

  auto* protonExcitation = new G4DNAExcitation("proton_G4DNAExcitation");
  auto* protonWaterExcitationModel = new G4DNAMillerGreenExcitationModel();
  protonWaterExcitationModel->SelectStationary(false);
  protonWaterExcitationModel->SetLowEnergyLimit(kExcWaterLow);
  protonWaterExcitationModel->SetHighEnergyLimit(kExcWaterHigh);
  protonExcitation->SetEmModel(protonWaterExcitationModel);

  auto* protonIceExcitationModel = new G4DNAEmfietzoglou_iceProtonExcitationModel();
  protonIceExcitationModel->SelectStationary(false);
  protonIceExcitationModel->SetLowEnergyLimit(kExcIceLow);
  protonIceExcitationModel->SetHighEnergyLimit(kExcIceHigh);
  protonExcitation->AddEmModel(2, protonIceExcitationModel);
  protonExcitation->SetMinKinEnergy(kExcWaterLow);
  protonExcitation->SetMaxKinEnergy(kExcIceHigh);
  ph->RegisterProcess(protonExcitation, G4Proton::ProtonDefinition());

  // Proton ionisation model tiling:
  // - Geant4-DNA water (Rudd): 10 eV - 100 keV
  // - Custom ice model:         100 keV - 10 MeV
  constexpr G4double kWaterLow = 10.0 * eV;
  constexpr G4double kWaterHigh = 100.0 * keV;
  constexpr G4double kIceLow = 100.0 * keV;
  constexpr G4double kIceHigh = 10.0 * MeV;

  auto* protonIonisation = new G4DNAIonisation("proton_G4DNAIonisation");
  auto* protonWaterModel = new G4DNARuddIonisationModel();
  protonWaterModel->SelectStationary(false);
  protonWaterModel->SetLowEnergyLimit(kWaterLow);
  protonWaterModel->SetHighEnergyLimit(kWaterHigh);
  protonIonisation->SetEmModel(protonWaterModel);

  auto* protonModel = new G4DNAEmfietzoglou_iceProtonIonisationModel();
  protonModel->SelectFasterComputation(false);
  protonModel->SelectStationary(false);
  protonModel->SetLowEnergyLimit(kIceLow);
  protonModel->SetHighEnergyLimit(kIceHigh);
  protonIonisation->AddEmModel(2, protonModel);
  protonIonisation->SetMinKinEnergy(kWaterLow);
  protonIonisation->SetMaxKinEnergy(kIceHigh);

  ph->RegisterProcess(protonIonisation, G4Proton::ProtonDefinition());
}

void PhysicsList_Proton::AddPhysics(const G4String&)
{
}

void PhysicsList_Proton::TrackingCut()
{
}

void PhysicsList_Proton::SetTrackingCut(G4bool)
{
}
