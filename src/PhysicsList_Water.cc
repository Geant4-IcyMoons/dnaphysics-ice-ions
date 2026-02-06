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
// * This  code  implementation is the result of the scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file PhysicsList_Water.cc
/// \brief Electron-only G4-DNA water physics list

#include "PhysicsList_Water.hh"

#include "G4EmDNAPhysics_option2.hh"
#include "G4EmParameters.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicsListHelper.hh"

#include "G4DNAElastic.hh"
#include "G4DNAExcitation.hh"
#include "G4DNAIonisation.hh"
#include "G4DNAVibExcitation.hh"
#include "G4DNAAttachment.hh"
#include "G4Electron.hh"
#include "G4DNABornExcitationModel1Tracked.hh"
#include "G4DNABornIonisationModel1Tracked.hh"
#include "G4DNAEmfietzoglouExcitationModelTracked.hh"
#include "G4DNAEmfietzoglouIonisationModelTracked.hh"

PhysicsList_Water::PhysicsList_Water() : G4VModularPhysicsList()
{
  SetDefaultCutValue(1.0 * micrometer);
  SetVerboseLevel(1);

  fEmPhysicsList = new G4EmDNAPhysics_option2();

  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(1 * eV, 1 * GeV);
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetMinEnergy(1 * eV);
  param->SetMaxEnergy(1 * GeV);
}

PhysicsList_Water::~PhysicsList_Water()
{
  delete fEmPhysicsList;
}

void PhysicsList_Water::ConstructParticle()
{
  fEmPhysicsList->ConstructParticle();
}

void PhysicsList_Water::ConstructProcess()
{
  AddTransportation();

  G4PhysicsListHelper* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  auto* electron = G4Electron::ElectronDefinition();

  auto* elastic = new G4DNAElastic("e-_G4DNAElastic_WATER");
  ph->RegisterProcess(elastic, electron);

  auto* vib = new G4DNAVibExcitation("e-_G4DNAVib_WATER");
  ph->RegisterProcess(vib, electron);

  auto* attachment = new G4DNAAttachment("e-_G4DNAAttachment_WATER");
  ph->RegisterProcess(attachment, electron);

  auto* excitation = new G4DNAExcitation("e-_G4DNAExcitation_WATER");
  auto* exc_emfi = new G4DNAEmfietzoglouExcitationModelTracked();
  exc_emfi->SetLowEnergyLimit(8. * eV);
  exc_emfi->SetHighEnergyLimit(10. * keV);
  excitation->SetEmModel(exc_emfi);
  auto* exc_born = new G4DNABornExcitationModel1Tracked();
  exc_born->SetLowEnergyLimit(10. * keV);
  exc_born->SetHighEnergyLimit(1. * MeV);
  excitation->AddEmModel(2, exc_born);
  ph->RegisterProcess(excitation, electron);

  auto* ionisation = new G4DNAIonisation("e-_G4DNAIonisation_WATER");
  auto* ion_emfi = new G4DNAEmfietzoglouIonisationModelTracked();
  ion_emfi->SetLowEnergyLimit(10. * eV);
  ion_emfi->SetHighEnergyLimit(10. * keV);
  ionisation->SetEmModel(ion_emfi);
  auto* ion_born = new G4DNABornIonisationModel1Tracked();
  ion_born->SetLowEnergyLimit(10. * keV);
  ion_born->SetHighEnergyLimit(1. * MeV);
  ionisation->AddEmModel(2, ion_born);
  ph->RegisterProcess(ionisation, electron);
}
